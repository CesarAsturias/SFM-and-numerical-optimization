import numpy as np
import ransac
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# @file PlaneModel.py
# @author Cesar
# @version 1.0
# Class to model a 3D point cloud to a plane using Ransac
# (http://www.scipy.org/CookBook/RANSAC)


class RansacModel(object):
    def __init__(self, debug=False):
        self.__debug = debug

    def fit(self, data):
        # Fit the a plane to the 3D points using the SVD decomposition (least
        # squares fit), i.e, minimizing the algebraic distance. The minimum
        # points necessaries to solve the system are 4.
        # @param data: 4 x 3 numpy array (x, y, z)
        [rows, cols] = data.shape
        p = np.ones((rows, 1))
        AB = np.hstack([data, p])  # Form (x y z 1) * (a b c d)^t = 0
        [u, d, v] = np.linalg.svd(AB, 0)
        B = v[3, :]
        norm = np.linalg.norm(B[0:3])
        B = B / norm
        return B

    def get_error(self, data, model):
        # Apply the model to all points, return the sum squared error per row
        # @param data: n x 4 numpy array (ex row (x y z 1))
        # @param model: 4 x 1 numpy array (parameters of the plane)
        # @return err_per_point: n x 1 error vector
        [rows, cols] = data.shape
        p = np.ones((rows, 1))
        AB = np.hstack([data, p])
        return np.dot(AB, model)


def test():
    # Generate input data
    mean = np.array([0.0, 0.0, 0.0])
    cov = np.array([[1.0, -0.5, 0.8], [-0.5, 1.1, 0.0], [0.8, 0.0, 1.0]])
    data = np.random.multivariate_normal(mean, cov, 50)

    print " Shape of generated data: {}".format(data.shape)

    # Create an instance of the RansacModel class and fit a plane to the noisy
    # data
    debug = False
    plane = RansacModel(debug)
    param = plane.fit(data)
    print "Plane : {}".format(param)

    # Create a regular grid to evaluate the fitted model
    X, Y = np.meshgrid(np.arange(-3.0, 3.0, 0.5), np.arange(-3.0, 3.0, 0.5))
    Z = (-param[0] * X - param[1] * Y - param[3]) / param[2]

    # Plot the result
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='r', s=50)
    plt.xlabel('X')
    plt.ylabel('Y')
    ax.set_zlabel('Z')
    ax.axis('equal')
    ax.axis('tight')
    plt.show()

    # Return error of the plane fitting
    print "Error: {}".format(np.mean(np.absolute(plane.get_error(data, param))))
    print "Error: {}".format(plane.get_error(data, param))

    # Now, use the RANSAC algorithm
    ransac_fit, ransac_data = ransac.ransac(data, plane, 4, 100, 1e-2, 10,
                                            debug=debug, return_all=True)
    # RANSAC returns a dictionary indicating the index of the inliers and
    # outliers. We can use it to draw them from the original data
    inliers = ransac_data['inliers']
    Z = (-ransac_fit[0] * X - ransac_fit[1] * Y - ransac_fit[3]) / ransac_fit[2]
    # outliers = ransac_data['outliers']
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)
    ax.scatter(data[inliers, 0], data[inliers, 1], data[inliers, 2], c='r',
               s=50)
    # ax.scatter(data[outliers, 0], data[outliers, 1], data[outliers, 2], c='b',
    #           s=50)
    plt.show()
    print "type of inliers: {}".format(type(data[inliers]))
    print "shape of inliers: {}".format(np.shape(data[inliers]))
    print "type of data: {}".format(type(data))
    print "shape of data: {}".format(np.shape(data))
    print "Error RANSAC: {}".format(np.mean(np.absolute(plane.get_error(data[inliers],
                                                               ransac_fit))))
    print "Error RANSAC: {}".format(plane.get_error(data[inliers],
                                                    ransac_fit))


if __name__ == '__main__':
    test()
