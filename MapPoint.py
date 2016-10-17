from scipy.spatial import distance

"""
.. codeauthor:: Cesar Gonzalez Gonzalez
:file MapPoint.py
"""


class MapPoint(object):
    """ The MapPoint class represents a 3D world point.

    .. _ORB-SLAM: https://arxiv.org/pdf/1502.00956.pdf
    .. _Hamming: https://en.wikipedia.org/wiki/Hamming_distance

    **Attributes**:

        .. data:: x_w

           Its 3D position :math:`\\mathbf{X}_{w,i}` in the *world* coordinate
           system.

        .. data:: ORB_descriptor

           Representative ORB descriptor :math:`D`, which is the associated
           ORB descriptor whose hamming distance is minimum with respect to all
           other associated descriptors in the keyframes in which the point is
           observed (see ORB-SLAM_).

        .. data:: descriptors

           List of descriptors associated with the image projection of the point
           in each frame. As the :py:class:`MapPoint.measured_2d_points`,
           this list is correlated with the connected_frames list.

           Each entry of this list is a numpy 1xd ndarray, where :math:`d` is
           the size of the descriptor (:math:`d=32` for ORB descriptors).

        .. data:: connected_frames

           List of camera index in which the point was observed.

        .. data:: measured_2d_points

           List of image KeyPoints for the 3D point. This list is correlated
           with the *connected_frames* attribute, i.e, the first KeyPoint is
           the 3D point as seen by the camera wich index is the first index in
           the *connected_frames* list.

    **Constructor**:

        The constructor can take 4 optional arguments:

            1. *coord* (Numpy 1x3 ndarray): The 3D coordinates of the point.
            2. *frame* (Integer): The index of the frame in which the point
               has been seen.
            3. *img_point* (Numpy 1x2 ndarray): 2D image coordinates of the
               point, as seen by the camera.
            4. *desc* (Numpy ndarray): Descriptor associated with the image
               point.

    """
    def __init__(self, coord=None, frame=None, img_point=None, desc=None):
        """ Constructor

        """
        if coord is not None:
            self.x_w = coord
        else:
            self.x_w = None
        if frame is not None:
            self.connected_frames = list(frame)
        else:
            self.connected_frames = []
        if img_point is not None:
            self.measured_2d_points = list(img_point)
        else:
            self.measured_2d_points = []
        if desc is not None:
            self.descriptors = list(desc)
            self.ORB_descriptor = desc
        else:
            self.descriptors = []
            self.ORB_descriptor = None

    def set_x(self, point_3d):
        """ Sets the 3D position :math:`\\mathbf{X}_{w,i}` of the map point

        :param point_3d: 3D coordinates to be used as point 3D.
        :type point_3d: Numpy 1x3 ndarray
        """
        self.x_w = point_3d

    def get_x(self):
        """ Returns the 3D coordinates of the point

        :returns: 3D coordinates of the point
        :rtype: Numpy 1x3 ndarray

        """
        return self.x_w

    def add_frame(self, frame):
        """ Adds a frame index to the list of connected frames.

        :param frame: The frame index
        :type frame: Integer

        """
        self.connected_frames.append(frame)

    def remove_frame(self, frame):
        """ Removes a frame index from the connected frames list and the
        2D coordinates and descriptor related with it.

        This method removes all occurrences of the frame index.

        :param frame: The frame index
        :type frame: Integer

        """
        index = [i for i, x in enumerate(self.connected_frames) if x == frame]
        # The next line could be used, but since we have extracted the
        # positions of the value we should use them because is easier
        # self.connected_frames = [x for x in self.connected_frames if x != frame]
        del self.connected_frames[index]
        del self.descriptors[index]
        del self.measured_2d_points[index]

    def add_image_point(self, point):
        """ Add a new projection of the map point in a frame.

        :param point: The 2D coordinates of the image point.
        :type point: Numpy 1x2 ndarray

        """
        self.measured_2d_points.append(point)

    def add_descriptor(self, desc):
        """ Adds a descriptor to the map point

        :param desc: Descriptor
        :type desc: Numpy ndarray (the shape will depend on the type of the
                    descriptor)
        """
        self.descriptors.append(desc)

    def add_observation(self, frame, point, desc):
        """ Adds a complete observation of the map point

        :param frame: Frame index
        :param point: 2D coordinates in the image plane
        :param desc: Associated descriptor
        :type frame: Integer
        :type point: Numpy 1x2 ndarray
        :type desc: Numpy ndarray
        """
        self.add_frame(frame)
        self.add_image_point(point)
        self.add_descriptor(desc)

    def compute_rep_desc(self):
        """ Computes the representative ORB descriptor of the map point.

        The representative ORB descriptor is computed using the following
        equation:

        .. math::

            D = \\min_i \\sum_j d_H(d_i, d_j)

        where :math:`d_H(d_i, d_j)` is the Hamming_ distance between the
        i-th descriptor and the j-th descriptor.

        """
        hamming_sum = 0
        hamming_list = []
        for i in range(len(self.descriptors)):
            for j in range(len(self.descriptors)):
                if j != i:
                    hamming_sum += distance.hamming(self.descriptors[i],
                                                    self.descriptors[j])
                else:
                    pass
            hamming_list.append(hamming_sum)
        index = hamming_list.index(min(hamming_list))
        self.ORB_descriptor = self.descriptors[index]
