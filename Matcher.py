import numpy as np
import cv2
import pdb

"""
.. codeauthor:: Cesar Gonzalez Gonzalez
: file Matcher.py

"""


class Matcher(object):
    """ The Matcher object allow us to find and store the keypoints of an image.

    We can use the ORB_ detector or the Flann_ detector. By default the object
    will use the ORB detector and the brute force matcher, but we can make
    use of the Flann-based matcher (with ORB or another detector) by
    passing to the object a dictionary with the detector and the matcher name.

    .. _ORB: http://www.willowgarage.com/sites/default/files/orb_final.pdf
    .. _Flann: http://docs.opencv.org/3.0-beta/doc/tutorials/features2d/feature_flann_matcher/feature_flann_matcher.html

    For Flann based matcher we need to pass two dictionaries which specifies the
    algorithm to be used, its related parameters, etc. First one is
    *IndexParams*. We can use one of the following algorithms:

        1. Linear brute force search: FLANN_INDEX_LINEAR (0)
        2. KDTree search. The matcher will construct a set of randomized
           kd-trees which will be searched in parallel: FLANN_INDEX_KDTREE(1)
        3. KMeans search. The matcher will construct an hierarchical k-means
           tree: FLANN_INDEX_KMEANS (2)

    **Attributes**:

        .. data:: detector

           The detector instance

        .. data:: matcher

           The matcher instance

        .. data:: flann_algorithm

           Dictionary containing the different possible algorithms for the
           Flann based matcher.

        .. data:: norm

           The norm to be used if the brute force matcher is used. Can be
           *L2* norm, if not using *ORB* detector or *Hamming* otherwise.

        .. data:: minkp

           Boolean. If True, the matching method have found at least one keypoint in
           both images.

        .. data:: minmatches

           Boolean. If True, there is at least one match between the keypoints.

        .. data:: kp1

           List of KeyPoints objects found by the detector in the first image.

        .. data:: kp2

           List of KeyPoints objects found by the detector in the second image.

        .. data:: desc1

           Numpy ndarray returned by the detector in the first image. This multi-
           dimensional array will have two dimensions:

               1. Array that stores the  descriptors:

                   .. code-block:: python

                      desc1[0] # Shows the first descriptor

                 We can get the number of descriptors returned by the detector:

                   .. code-block:: python

                      n_desc = (np.shape(desc1)[0])

               2. The descriptor itself. In the case of the *ORB* descriptor
                  it will be of size 32 (see the original paper of ORB_).



    **Constructor**:

        The constructor takes as argument a dictionary that contains the
        following keys:

            1. detector: Can be 'orb', 'surf' or 'sift'.

            2. matcher: Can be 'brute_force' or 'flann'.

            3. flann_index_params: The Flann based matcher takes different
            parameters depending on the used algorithm. This argument should
            be a dictionary containing the following keys:

                1. *LSH* algorithm:

                    a. algorithm: The algorithm to be used (*FLANN_LSH*)
                    b. table_number: The number of hash tables used (between
                       10 and 30 usually, but in the documentation they said
                       that 6 usually works better.
                    c. key_size: The size of the hash key in bits (between
                       10 and 20 usually)
                    d. multi_probe_level: The number of bits to shift to check
                       for neighboring buckets (0 is regular LSH, 2 is recommen-
                       ded).

                2. *KDTREE* algorithm:

                    a. algorithm: The algorithm to be used (*FLANN_KDTREE*)
                    b. trees: The number of parallel kd-trees to use. Good
                       values are in the range [1..16].

                :example:

                    >>> flann_index_params = dict(algorithm='FLANN_LSH',
                                                  table_number=6,
                                                  key_size=12,
                                                  multi_probe_level=1)
                    >>> flann_index_params = dict(algorithm='FLANN_KDTREE',
                                                  trees=5)

            4. flann_search_params: The Flann based matcher performs a K-nearest
               neighbor search for a given query point using the index. This
               search use a *checks* parameter that indicates the number of
               times the tree(s) in the index should be recursively traversed.
               A higher value for this parameter would give better search
               precision, but also take more time. This parameter is optional,
               so if we don't know which value should it takes then we can
               also pass an empty dictionary.

               :example:

                   >>> flann_search_params = dict(checks=50)




    """
    def __init__(self, parameters):
        # Initiate detector and matcher
        if parameters['detector'] == 'orb':
            self.detector = cv2.ORB_create(400)
            self.norm = cv2.NORM_HAMMING
        elif parameters['detector'] == 'surf':
            self.detector = cv2.xfeatures2d.SURF_create(800)
            self.norm = cv2.NORM_L2
        elif parameters['detector'] == 'sift':
            self.detector = cv2.xfeatures2d.SIFT_create()
            self.norm = cv2.NORM_L2
        else:
            raise KeyError("Valid detectors: orb, surf, sift")
        if parameters['matcher'] == 'flann':
            self.matcher = cv2.FlannBasedMatcher(parameters['flann_index_params'],
                                                 parameters['flann_search_params'])
        else:
            # Use the brute force matcher. We set the *crossCheck* parameter to
            # False because we are going to check the matches ourselves.
            self.matcher = cv2.BFMatcher(self.norm, crossCheck=False)

        self.kp1 = []
        self.kp2 = []
        self.desc1 = None
        self.desc2 = None
        self.matches = None
        self.ratio = 0.65
        self.matches1 = None
        self.matches2 = None
        self.good_matches = None
        self.good_kp1 = []
        self.good_kp2 = []
        self.n_matches = 0
        # The following variables are used to work with numpy functions
        self.global_matches = []  # Numpy array
        self.global_kpts1 = []
        self.global_kpts2 = []
        # Create lists where we store the keypoints in their original format for
        # future uses. Also, store the descriptors
        self.curr_kp = []  # List of keypoints
        self.prev_kp = []
        self.curr_dsc = []  # List of descriptors
        self.prev_dsc = []
        self.is_minkp = None
        self.is_minmatches = None
        self.lk_params = dict(winSize=(15, 15),
                              maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS |
                                        cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    def match(self, img_new, img_prev):
        # Compute the matches for the two images using the Brute Force matcher
        # @param img_new: The current image
        # @param img_prev: The reference image
        # Initialize control variables
        self.is_minkp = None
        self.is_minmatches = None
        self.good_kp1 = []
        self.good_kp2 = []
        self.good_matches = None
        self.kp1, self.desc1 = self.orb.detectAndCompute(img_prev, None)
        self.kp2, self.desc2 = self.orb.detectAndCompute(img_new, None)

        # Check descriptors

        # There are any keypoints?
        if not self.kp1 or not self.kp2:
            self.is_minkp = False
            print "The list of keypoints is emtpy"
        else:
            print "The list of keypoints is NOT empty"
            if np.size(self.desc1) == 0 or np.size(self.desc2) == 0:
                print "NO DESCRIPTORS"
                self.is_minkp = False
            else:
                self.is_minkp = True

        # Store the keypoints
        # print "len original", len(self.kp1)
        # for i in range(len(self.kp1)):
        #     self.curr_kp.append(self.kp1[i])
        # for i in range(len(self.kp2)):
        #     self.prev_kp.append(self.kp2[i])

        if self.is_minkp:
            print " Matching..."
            print self.is_minkp
            self.matches1 = self.bf.knnMatch(self.desc1, self.desc2, 2)
            self.matches2 = self.bf.knnMatch(self.desc2, self.desc1, 2)

            # Check now if there are any matches
            if self.matches1 is None or self.matches2 is None:
                print "There are keypoints, but no one matches"
            else:
                self.is_minmatches = True
                self.good_matches = self.filter_matches(self.matches1,
                                                        self.matches2)
        # self.matches = sorted(self.matches, key=lambda x: x.distance)

    def filter_distance(self, matches):
        # Filter the matches based on a distance threshold
        # @param matches: List of matches (matcher objects)
        # pdb.set_trace()
        # Clear matches for wich NearestNeighbor (NN) ratio is > than threshold
        dist = []
        sel_matches = []
        thres_dist = 0
        temp_matches = []
        # print "Entering in the filter_distance function..."

        for i in range(0, len(matches) - 1):
            # We keep only those match objects with two matches:
            if (len(matches[i])) == 2:
                # print "There is at least one match with 2 candidate points"
                # If there are two matches:
                for j in range(0, 2):
                    dist.append(matches[i][j].distance)
                    temp_matches.append(matches[i])

        # Now, calculate the threshold:
        if dist:
            thres_dist = (sum(dist) / len(dist)) * self.ratio
        else:
            return

        # Keep only reasonable matches based on the threshold distance:
        for i in range(0, len(temp_matches)):
            if (temp_matches[i][0].distance / temp_matches[i][1].distance) < thres_dist:

                sel_matches.append(temp_matches[i])
        print "matches after distance", len(sel_matches)

        return sel_matches

    def filter_asymmetric(self, matches1, matches2):
        # Filter the matches with the symetrical test.
        # @param matches1: First list of matches
        # @param matches2: Second list of matches
        # @return sel_matches: filtered matches
        # Keep only symmetric matches
        sel_matches = []
        # For every match in the forward direction, we remove those that aren't
        # found in the other direction
        for match1 in matches1:
            for match2 in matches2:
                # If matches are symmetrical:
                    if (match1[0].queryIdx) == (match2[0].trainIdx) and \
                         (match2[0].queryIdx) == (match1[0].trainIdx):
                        # We keep only symmetric matches and store the keypoints
                        # of this matches
                        sel_matches.append(match1)
                        self.good_kp2.append(self.kp2[match1[0].trainIdx].pt)
                        self.good_kp1.append(self.kp1[match1[0].queryIdx].pt)
                        # Store also the keypoints in original form
                        self.curr_kp.append(self.kp1[match1[0].queryIdx])
                        self.prev_kp.append(self.kp2[match1[0].trainIdx])
                        self.curr_dsc.append(self.desc1[match1[0].queryIdx])
                        self.prev_dsc.append(self.desc2[match1[0].trainIdx])
                        break

        # We have stored twice every keypoint. Remove them

        self.good_kp1 = self.good_kp1[::2]
        self.good_kp2 = self.good_kp2[::2]
        sel_matches = sel_matches[::2]
        print "matches after simmetric filter", len(sel_matches)

        return sel_matches

    def filter_matches(self, matches1, matches2):
        # This function filter two list of matches based on the distance
        # threshold and the symmetric test
        # @param matches1: First list of matches
        # @param matches2: Second list of matches
        # @return good_matches: List of filtered matches

        matches1 = self.filter_distance(matches1)
        if matches1:
            print ("Matches1 after distance filter:\
                    {}".format(len(matches1)))
        matches2 = self.filter_distance(matches2)
        if matches2:
            print ("Matches2 after distance filter:\
                     {}".format(len(matches2)))
        print "Ended distance filtering"
        if not matches1 or not matches2:
            print "Not matches1 or not matches2"
            self.is_minmatches = False
        else:
            self.is_minmatches = True
        print self.is_minmatches
        if self.is_minmatches:
            print "Go to filter asymmetric matches..."
            good_matches = self.filter_asymmetric(matches1, matches2)
            return good_matches
        else:
            return None

    def match_flann(self, img_new, img_prev):
        # Compute matches for the two images based on Flann.
        # @param img_new: Current frame
        # @param img_prev: Reference frame
        # First, keypoints and descriptors for both images
        self.kp1, self.desc1 = self.orb.detectAndCompute(img_new, None)
        self.kp2, self.desc2 = self.orb.detectAndCompute(img_prev, None)
        # Next, match:
        print "kp1",  len(self.kp1)
        print "kp2", len(self.kp2)
        if self.kp1 and self.kp2:
            matches1 = self.flann_matcher.knnMatch(self.desc1, self.desc2, k=2)
            print "matches1", len(matches1)

            matches2 = self.flann_matcher.knnMatch(self.desc2, self.desc1, k=2)
            print "matches2", len(matches2)

            self.good_matches = self.filter_matches(matches1, matches2)
        else:
            print "No matches found"

    def draw_matches(self, img, matches):
        # Draw matches in the last image
        # @param img: image
        # @param matches: a matcher object (opencv)
        # @param kp1: keypoints of the old frame
        # @param kp2: keypoints of the new frame
        # @return img: image with lines between correlated points
        for i in range(len(matches)):
            idtrain = matches[i][0].trainIdx
            idquery = matches[i][0].queryIdx

            point_train = self.kp2[idtrain].pt
            point_query = self.kp1[idquery].pt

            point_train = self.transform_float_int_tuple(point_train)
            point_query = self.transform_float_int_tuple(point_query)

            cv2.line(img, ((point_train[0]), (point_train[1])),
                     ((point_query[0]), (point_query[1])), (255, 0, 0))

        return img

    def draw_matches_np(self, img, kpts_c, kpts_p):
        # Draw the matches in the image img, taking as input a numpy array
        # @param img: image
        # @param kpts_c: keypoints in the current image (numpy ndarray)
        # @param kpts_p: keypoints in the previous image (numpy ndarray)
        # @return img: image with lines between correlated points

        for i in range(len(kpts_c)):

            cv2.line(img, ((kpts_c[i][0]), (kpts_c[i][1])),
                     ((kpts_p[i][0]), (kpts_p[i][1])), (255, 0, 0))

        return img

    def draw_outliers_np(self, img, kpts_c, kpts_p):
        # Draw the matches in the image img, taking as input a numpy array
        # @param img: image
        # @param kpts_c: keypoints in the current image (numpy ndarray)
        # @param kpts_p: keypoints in the previous image (numpy ndarray)
        # @return img: image with lines between correlated points

        for i in range(len(kpts_c)):

            cv2.line(img, ((kpts_c[i][0]), (kpts_c[i][1])),
                     ((kpts_p[i][0]), (kpts_p[i][1])), (0, 0, 255))

        return img

    def transform_float_int_tuple(self, input_tuple):
        output_tuple = [0, 0]
        if not input_tuple is None:
            for i in range(0, len(input_tuple)):
                output_tuple[i] = int(input_tuple[i])
        else:
            return input_tuple

        return output_tuple

    def append_matches(self):
        # Store all matches in one list

        for i in range(len(self.good_matches)):

            self.global_matches.append(self.good_matches[i])

    def append_keypoints1(self):
        # Store keypoints from the current image in one list

        for i in range(len(self.good_kp1)):

            self.global_kpts1.append(self.good_kp1[i])

    def append_keypoints2(self):
        # Store keypoints from the current image in one list

        for i in range(len(self.good_kp2)):

            self.global_kpts2.append(self.good_kp2[i])

    def append_global(self):

        self.append_keypoints1()
        self.append_keypoints2()
        self.append_matches()

    def sum_coord(self, x_ini, y_ini):
        # In order to get the true coordinates of the keypoits distributed along
        # over the entire image, not the coordinates refered to the roi's, we
        # have to sum the start vector to every keypoint

        for i in range(len(self.good_kp1)):

            self.good_kp1[i] = list(self.good_kp1[i])

            self.good_kp2[i] = list(self.good_kp2[i])

            self.good_kp1[i][0] += x_ini

            self.good_kp1[i][1] += y_ini

            self.good_kp2[i][0] += x_ini

            self.good_kp2[i][1] += y_ini

            self.good_kp1[i] = np.array([self.good_kp1[i][0],
                                         self.good_kp1[i][1]]).reshape(2)

            self.good_kp2[i] = np.array([self.good_kp2[i][0],
                                         self.good_kp2[i][1]]).reshape(2)

    def lktracker(self, img_prev, img_curr, prev_points):
            # Tracks the prev_points in the current image.
            # @param img_prev: image, the previous image
            # @param img_curr: the current image
            # @param prev_points: vector of points in the  previous image
            list_tracks = []
            curr_points, st, err = cv2.calcOpticalFlowPyrLK(img_prev,
                                                            img_curr,
                                                            prev_points,
                                                            None,
                                                            **self.lk_params)
            print "LK current points", len(curr_points)

            prev_points2, st, err = cv2.calcOpticalFlowPyrLK(img_curr,
                                                             img_prev,
                                                             curr_points,
                                                             None,
                                                             **self.lk_params)
            print "LK prev poiints", len(prev_points)
            d = abs(prev_points - prev_points2).reshape(-1, 2).max(-1)
            print "d", d
            good = d < 2
            # print "good", good
            for (x, y), good_flag in zip(curr_points.reshape(-1, 2), good):
                if not good_flag:
                    continue
                list_tracks.append((x, y))

            return good, prev_points2, list_tracks

    #def track(self, img, match):
        # This function implements the tracking of the points across five images
        # or less if we loss too much points.
