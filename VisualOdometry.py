import numpy as np
import cv2
import threading
from multiprocessing import Process, Queue
from scipy import linalg
from scipy import optimize
from math import cos, sin, exp

# @file VisualOdometry.py
# @author Cesar
# @version 1.0
# Class VisualOdometry. This class contain the measured
# odometry. Implements the calculus of the odometry based on two consecutive
# images.


class VisualOdometry(object):
    def __init__(self):
        self.F = None
        self.inlier_points_new = None
        self.inlier_points_prev = None
        self.outlier_points_new = None
        self.outlier_points_prev = None
        self.mask = None
        self.H = None
        self.maskH = None
        self.e = None  # epipole
        self.cam1 = Camera(None)
        self.cam2 = Camera(None)
        self.correctedkpts1 = None
        self.correctedkpts2 = None  # Feature locations corrected by the optimal
        # triangulation algorithm. This are the estimated features in the Gold
        # Standard algorithm for estimating F (H&Z page 285)
        self.K = np.array([[7.18856e+02, 0.0, 6.071928e+02],
                          [0.0, 7.18856e+02, 1.852157e+02],
                          [0.0, 0.0, 1.0]])  # Calibration matrix
        self.height = 1.65  # Height of the cameras in meters
        self.E = None  # Essential matrix
        self.maskE = None  # Mask for the essential matrix

        self.structure = None  # List  of 3D points (triangulated)

    def FindFundamentalRansac(self, kpts1, kpts2):
        # Compute Fundamental matrix from a set of corresponding keypoints,
        # within a RANSAC scheme
        # @param kpts1: list of keypoints of the previous frame
        # @param kpts2: list of keypoints of the current frame

        kpts1 = np.float32(kpts1)
        kpts2 = np.float32(kpts2)
        self.F, self.mask = cv2.findFundamentalMat(kpts1, kpts2, cv2.FM_RANSAC)

    def FindEssentialRansac(self, kpts1, kpts2):
        # Compute Essential matrix from a set of corresponding points
        # @param kpts1: list of keypoints of the previous frame
        # @param kpts2: list of keypoints of the current frame

        kpts1 = np.float32(kpts1)
        kpts2 = np.float32(kpts2)

        # findEssentialMat takes as arguments, apart from the keypoints of both
        # images, the focal length and the principal point. Looking at the
        # source code of this function
        # (https://github.com/Itseez/opencv/blob/master/modules/calib3d/src/five-point.cpp)
        # I realized that these parameters are feeded to the function because it
        # internally create the camera matrix, so they must be in pixel
        # coordinates. Hence, we take them from the already known camera matrix:

        focal = 3.37
        pp = (2.85738, 0.8681)

        # pp = (self.K[0][2], self.K[1][2])

        self.E, self.maskE = cv2.findEssentialMat(kpts2, kpts1, focal, pp,
                                                  cv2.RANSAC, 0.999, 1.0,
                                                  self.maskE)

    def FindFundamentalRansacPro(self, queue):
        # Compute Fundamental matrix from a set of corresponding keypoints,
        # within a RANSAC scheme
        # @param kpts1: list of keypoints of the previous frame
        # @param kpts2: list of keypoints of the current frame

        temp = queue.get()
        kpts1 = temp[0]
        kpts2 = temp[1]
        F = temp[2]

        F, mask = cv2.findFundamentalMat(kpts1, kpts2, cv2.FM_RANSAC)
        res = [F, mask]
        queue.put(res)

    def EstimateF_multiprocessing(self, kpts1, kpts2):
        # Estimate F using the multiprocessing module
        # @param kpts1: list of keypoints of the previous (first) frame
        # @param kpts2: list of keypoints of the current frame
        # @return kpts1, kpts2: Inliers

        kpts1 = np.float32(kpts1)
        kpts2 = np.float32(kpts2)

        data_queued = [kpts1, kpts2, self.F]
        q = Queue()
        q.put(data_queued)
        # Compute F in a parallel process:

        p = Process(target=self.FindFundamentalRansacPro,
                    args=(q, ))
        p.start()
        # We must wait until the process has finished because then the queue
        # will be filled with the result. Otherwise, we can't use the get method
        # (FIFO model of queue)

        p.join()
        res = q.get()
        self.F = res[0]
        self.mask = res[1]

        # Select only inlier points

        self.outlier_points_new = [kpts1[self.mask.ravel() == 0]]

        self.outlier_points_prev = [kpts2[self.mask.ravel() == 0]]

        self.outlier_points_new = np.float32(self.outlier_points_new[0])

        self.outlier_points_prev = np.float32(self.outlier_points_prev[0])

        return [kpts1[self.mask.ravel() == 1],
                kpts2[self.mask.ravel() == 1]]

    def EstimateF_multithreading(self, kpts1, kpts2):
        t = threading.Thread(target=self.FindFundamentalRansac,
                             args=(kpts2, kpts1, ))
        t.start()

    def FindHomographyRansac(self, kpts1, kpts2):
        # Find the homography between two images given corresponding points
        kpts1 = np.float32(kpts1)
        kpts2 = np.float32(kpts2)

        self.H, self.maskH = cv2.findHomography(kpts1, kpts2, cv2.RANSAC, 1.0)

    def get_epipole(self, F):
        # Return the (right) epipole from a fundamental matrix F.
        # Use with F.T for left epipole
        # @param F: Fundamental Matrix (numpy 3x3 array)

        # Null space of F (Fx = 0)

        U, S, V = linalg.svd(F)
        self.e = V[-1]
        self.e = self.e / self.e[2]

    def skew(self, a):
        # Return the matrix A such that a x v = Av for any v
        # @param a: numpy vector

        return np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])

    def P_from_F(self, F):
        # Computes the second camera matrix (assuming P1 = [I 0] from a
        # fundamental matrix.
        # @param F: Numpy matrix (Fundamental)

        F = self.F.T

        self.get_epipole(F)  # Left epipole

        Te = self.skew(self.e)
        self.cam2.set_P(np.vstack((np.dot(Te, F.T).T, self.e)).T)

    def create_P1(self):
        # Initialize P1 = [I | 0]

        self.cam1.set_P(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]))

    def optimal_triangulation(self, kpts1, kpts2):
        # For each given point corresondence points1[i] <-> points2[i], and a
        # fundamental matrix F, computes the corrected correspondences
        # new_points1[i] <-> new_points2[i] that minimize the geometric error
        # d(points1[i], new_points1[i])^2 + d(points2[i], new_points2[i])^2,
        # subject to the epipolar constraint new_points2^t * F * new_points1 = 0
        # Here we are using the OpenCV's function CorrectMatches.

        # @param kpts1 : keypoints in one image
        # @param kpts2 : keypoints in the other image
        # @return new_points1 : the optimized points1
        # @return new_points2 : the optimized points2

        # First, we have to reshape the keypoints. They must be a 1 x n x 2
        # array.

        kpts1 = np.float32(kpts1)  # Points in the first camera
        kpts2 = np.float32(kpts2)  # Points in the second camera

        # 3D Matrix : [kpts1[0] kpts[1]... kpts[n]]

        pt1 = np.reshape(kpts1, (1, len(kpts1), 2))

        pt2 = np.reshape(kpts2, (1, len(kpts2), 2))

        new_points1, new_points2 = cv2.correctMatches(self.F, pt2, pt1)

        self.correctedkpts1 = new_points1
        self.correctedkpts2 = new_points2

        # Transform to a 2D Matrix: 2xn

        kpts1 = (np.reshape(new_points1, (len(kpts1), 2))).T
        kpts2 = (np.reshape(new_points2, (len(kpts2), 2))).T

        print np.shape(kpts1)

        points3D = cv2.triangulatePoints(self.cam1.P, self.cam2.P, kpts2, kpts1)

        self.structure = points3D / points3D[3]  # Normalize points [x, y, z, 1]

        array = np.zeros((4, len(self.structure[0])))

        for i in range(len(self.structure[0])):

            array[:, i] = self.structure[:, i]

        self.structure = array

        # The individual points are selected like these:

        # self.structure[:, i]. It's a 4 x n matrix

    def opt_triangulation(self, x1, x2, P1, P2):
        # For each given point corresondence points1[i] <-> points2[i], and a
        # fundamental matrix F, computes the corrected correspondences
        # new_points1[i] <-> new_points2[i] that minimize the geometric error
        # d(points1[i], new_points1[i])^2 + d(points2[i], new_points2[i])^2,
        # subject to the epipolar constraint new_points2^t * F * new_points1 = 0
        # Here we are using the OpenCV's function CorrectMatches.

        # @param x1: points in the first camera, list of vectors x, y
        # @param x2: points in the second camera
        # @param P1: Projection matrix of the first camera
        # @param P2: Projection matrix of the second camera
        # @return points3d: Structure of the scene, 3 x n matrix

        x1 = np.float32(x1)  # Imhomogeneous
        x2 = np.float32(x2)

        # 3D Matrix : [kpts1[0] kpts[1]... kpts[n]]

        x1 = np.reshape(x1, (1, len(x1), 2))

        x2 = np.reshape(x2, (1, len(x2), 2))

        self.correctedkpts1, self.correctedkpts2 = cv2.correctMatches(self.F,
                                                                      x1, x2)
        # Now, reshape to n x 2 shape
        self.correctedkpts1 = self.correctedkpts1[0]
        self.correctedkpts2 = self.correctedkpts2[0]
        # and make homogeneous
        x1 = self.make_homog(np.transpose(self.correctedkpts1))
        x2 = self.make_homog(np.transpose(self.correctedkpts2))

        # Triangulate
        # This function needs as arguments the coordinates of the keypoints
        # (form 3 x n) and the projection matrices

        points3d = self.triangulate_list(x1, x2, P2, P1)

        self.structure = points3d  # 3 x n matrix

        return points3d

    def triangulate_point(self, x1, x2, P2, P1):
        # Point pair triangulation from least squares solution
        M = np.zeros((6, 6))
        M[:3, :4] = P1
        M[3:, :4] = P2
        M[:3, 4] = -x1
        M[3:, 5] = -x2

        U, S, V = linalg.svd(M)
        X = V[-1, :4]

        return X / X[3]

    def triangulate_list(self, x1, x2, P1, P2):
        # Two view triangulation of points in homogeneous coordinates (several)

        n = x1.shape[1]
        if x2.shape[1] != n:
            raise ValueError("Number of points don't match")

        X = [self.triangulate_point(x1[:, i],
                                    x2[:, i], P1, P2) for i in range(n)]
        return np.array(X).T

    def make_homog(self, points):
        # Convert to homogeneous coordinates
        return np.vstack((points, np.ones((1, points.shape[1]))))

    def triangulate(self, kpts1, kpts2):

        kpts1 = (np.reshape(kpts1, (len(kpts1), 2))).T
        kpts2 = (np.reshape(kpts2, (len(kpts2), 2))).T

        points3D = None

        points3D = cv2.triangulatePoints(self.cam1.P, self.cam2.P, kpts1, kpts2)

        points3D = points3D / points3D[3]

        return points3D

    def convert_from_homogeneous(self, kpts):
        # Convert homogeneous points to euclidean points
        # @param kpts: List of homogeneous points
        # @return pnh: list of euclidean points

        # Remember that every function in OpenCV need us to specify the data
        # type. In addition, convertPointsFromHomogeneous needs the shape of the
        # arrays to be correct. The function takes a vector of points in c++
        # (ie. a list of several points), so in numpy we need a multidimensional
        # array: a x b x c where a is the number of points, b=1, and c=2 to
        # represent 1x2 point data.

        if len(kpts[0]) == 3:

            for i in range(len(kpts)):

                kpts[i].reshape(-1, 1, 3)

                kpts[i] = np.array(kpts[i], np.float32).reshape(-1, 1, 3)

            pnh = [cv2.convertPointsFromHomogeneous(x) for x in kpts]

            for i in range(len(pnh)):

                pnh = np.array(pnh[i], np.float32).reshape(1, 2, 1)

        elif len(kpts[0]) == 4:

            for i in range(len(kpts)):

                kpts[i].reshape(-1, 1, 4)

                kpts[i] = np.array(kpts[i], np.float32).reshape(-1, 1, 4)

            pnh = [cv2.convertPointsFromHomogeneous(x) for x in kpts]

            for i in range(len(pnh)):

                pnh[i] = np.array(pnh[i], np.float32).reshape(1, 3, 1)

        elif len(kpts) == 3:

            pnh = np.zeros((2, len(kpts[0])))

            for i in range(len(kpts[0])):

                pnh[:, i] = kpts[:2, i]

        return pnh

    def convert_array2d(self, kpts):

        print len(kpts[:, 0])

        a = np.zeros((len(kpts[:, 0]), 2))

        for i in range(len(kpts[:, 0])):

            a[i, :] = kpts[i, :2]

        return a

    def functiontominimize(self, params, x1, x2):
        # This is the function that we will pass to the levenberg-marquardt
        # algorithm. It extracts the camera matrix P' from the params vector
        # (params[0:8]). Then, it triangulate the 3D points
        # X_hat and obtain the estimated x and x'. Finally, it returns the
        # reprojection error (not squared).

        # @param x1: inliers in the previous image, i.e, the measured points in
        # the previous image --> 3 x n array
        # @param x2: inliers in the second image
        # @param params: the parameter vector to minimize. It contains the two
        # projection matrices and the structure of the scene
        # @return e_rep: reprojection error

        if self.cam1 is None:
            self.create_P1()

        # Obtain the second camera matrix from the params vector
        P2 = None
        P2 = params[0:12].reshape(3, 4)
        p = params[12:len(params)]
        l = np.shape(p)
        # Obtain the structure matrix from param vector
        X = np.reshape(p, (3, l[0] / 3))
        # Make homogeneous
        X = self.make_homog(X)

        # Project the structure to both images and find residual:
        self.cam2.set_P(P2)

        x1_est = None
        x2_est = None

        x1_est = self.cam1.project(X)  # The estimated projections
        x2_est = self.cam2.project(X)  # 3 x n

        error_image1 = self.residual(x1, x1_est).ravel()
        error_image2 = self.residual(x2, x2_est).ravel()
        error = np.append(error_image1, error_image2)

        return error

    def residual(self, x1, x2):
        # Reprojection error. This function compute the squared distance between
        # all the correlated points in the x1 and x2 arrays, which are expected
        # to be numpy arrays.

        # @param x1: numpy nx2 array
        # @param x2: numpy 3 x n  array
        # @return squared euclidean distance

        # Since we x2 is a 3 x n array we have to proceed with a for loop.
        # TODO: Make code for converting automaticaly the shape of the arrays
        # tho the desired one. Perhaps, this could be done in the calling
        # function

        a, b = np.shape(x2)
        error = np.zeros((b, 1))

        x2_temp = np.delete(x2, 2, 0).transpose()  # Delete last row
        error = np.subtract(x1, x2_temp)  # subtract [x - x_prime, y - y_prime]

        return error  # numpy n  ndarray n x 2 (

    def optimize_F(self, x1, x2):
        # Wrapper for the optimize.leastsq function.

        # Transform camera matrix into vector:
        vec_P2 = None
        vec_P2 = np.hstack(self.cam2.P)

        # Transform the structure (matrix 3 x n) to 1d vector
        vec_str = None
        vec_str = np.delete(self.structure, 3, 0)  # The ones aren't params
        vec_str = vec_str.reshape(-1)

        param = vec_P2
        param = np.append(param, vec_str)

        # Pass them, and additional arguments to leastsq function.
        # TODO: redefine error function and create params vector

        param_opt, param_cov = optimize.leastsq(self.functiontominimize,
                                                param, args=(x1, x2))

        return param_opt, param_cov

    def recover_structure(self, vector_param):
        # Recover the structure matrix from the optimized vector of parameters
        # @param vector_param: vector of optimized parameters
        p = vector_param[12:len(vector_param)]
        l = np.shape(p)
        self.structure = np.reshape(p, (3, l[0] / 3))
        return self.structure

    def E_from_F(self):
        #
        # Get the essential matrix from the fundamental
        print np.shape(self.K.transpose())
        print np.shape(self.F)
        self.E = self.K.transpose().dot(self.F).dot(self.K)

    def get_pose(self, curr_kpts, prev_kpts, focal, pp):
        # Recover the rotation matrix and the translation vector from the
        # essential matrix E.
        # @param curr_kpts: Keypoints of the current frame (np.array float)
        # @param prev_kpts: Keypoints of the previous frame
        # @param focal: focal lenght of the camera
        # @param pp: principal point of the camera
        # @param mask: mask , it will store the mask with the points used to
        # recover the pose. Use this mask to filter the 3D points to be used in
        # following operations.
        # @return R: Rotation matrix, to be calculated
        # @return t: translation vector, to be calculated

        points, R, t, mask = cv2.recoverPose(self.E, curr_kpts, prev_kpts,
                                             focal, pp)
        return R, t

    def compute_scale(self, plane_model, scene):
        # Compute the scale of the scene based on a plane fitted to the 3D point
        # cloud represented by scene. The plane_model is fitted using a
        # least-squares approach inside a RANSAC scheme (see PlaneModel.py)
        # @param plane_model: the parameters of the plane (numpy array)
        # @param scene: 3D points marked as inliers by the RANSAC algorithm in
        # the process of estimating the plane. (4 x n) numpy array
        # @return scale: scale of the scene (float)

        # First compute the distance for every inlier and take the mean as the
        # final distance
        distance_sum = 0
        for i in range(np.shape(scene)[1]):
            distance = (np.dot(plane_model, scene[:, i])) / \
                        np.linalg.norm(plane_model)
            distance_sum += distance
        # Compute the mean distance and the corresponding scale as H / d
        mean = distance_sum / np.shape(scene)[1]
        scale = self.height / mean

        return scale

    def compute_scale2(self,  scene, pitch=0):
        # Compute the scale using an heuristic approach. For every triangulated
        # point compute it's height and the height difference with respect to
        # the other points. Sum all this differences and use a heuristic
        # function to decide which height is selected
        # @param pitch: The pitch angle of the camera (by default zero)
        # @param scene: 3D points of the hypothetical ground plane (4 x n)
        max_sum = 0
        for i in range(np.shape(scene)[1]):
            h = scene[1][i] * cos(pitch) - scene[2][i] * sin(pitch)
            height_sum = 0
            for j in range(np.shape(scene)[1]):
                h_j = scene[1][j] * cos(pitch) - scene[2][j] * sin(pitch)
                height_diff = h_j - h
                height_sum += exp(-50 * height_diff * height_diff)
            if height_sum > max_sum:
                max_sum = height_sum
                best_idx = i
        scale = scene[1][best_idx] * cos(pitch) - \
                scene[2][best_idx] * sin(pitch)
        return scale


class Camera(object):

    def __init__(self, P):
        self.P = P
        self.K = np.array([[7.18856e+02, 0.0, 6.071928e+02],
                          [0.0, 7.18856e+02, 1.852157e+02],
                          [0.0, 0.0, 1.0]])
        self.pp = np.array([self.K[0, 2], self.K[1, 2]])
        self.focal = self.K[0, 0]
        self.R = None
        self.t = None
        self.c = None  # Camera center

    def project(self, X):

        x = np.dot(self.P, X)

        for i in range(3):

            x[i] /= x[2]

        return x

    def set_P(self, P):
        # Set camera matrix
        self.P = P

    def set_K(self, K):

        self.K = K

    def compute_P(self, R, t):
        # Compute the camera matrix given the extrinsic parameters R and t
        # @param R: Rotation matrix
        # @param t: translation vector
        pose = np.column_stack((R, t))
        self.P = np.dot(self.K, pose)
