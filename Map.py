"""
.. codeauthor:: Cesar Gonzalez Gonzalez
:file Map.py
"""


class Map(object):
    """ The Map class represents the world map as seen by the camera.

    **Attributes**:

        .. data:: cameras

           List of Camera objects. One Camera per frame.

           .. seealso::

               :py:mod:`Camera.Camera`

        .. data:: structure

           List of 3D points, the structure reconstruction. Each 3D point is
           an instance of the :py:mod:`MapPoint.MapPoint` class.

           .. seealso::

               :py:mod:`MapPoint.MapPoint`

    **Constructor**:

        We can pass to the constructor a camera object (the initial frame) and
        a list of 3D MapPoint objects that represents the structure seen at
        the moment. If don't pass parameters to the constructor, then the
        first Camera (Frame) will be set as the origin (no rotation and no
        translation), and the structure will be an empty list.

    """
    def __init__(self, init_cam=None, init_structure=None):
        """ Constructor

        :param init_cam: The initial camera (optional)
        :param init_structure: The initial structure (optional)
        :type init_cam: :py:class:`Camera.Camera`
        :type init_structure: List of :py:class:`MapPoint` objects.

        """
        self.cameras = []
        self.structure = []
        if init_cam is not None:
            self.add_camera(init_cam)
        if init_structure is not None:
            self.add_list_mappoints(init_structure)

    def add_camera(self, camera):
        """ Add a camera (frame) to the list of cameras in *cameras* attribute.

        :param camera: The new camera (frame).
        :type camera: :py:class:`Camera.Camera` object

        """
        self.cameras.append(camera)

    def add_list_mappoints(self, mappoints):
        """ Add a list of mappoints to the list of mappoints in *structure*
        attribute.

        :param mappoints: The list of mappoints
        :type mappoints: List of :py:class:`MapPoint.MapPoint` objects

        """
        self.structure.extend(mappoints)

    def add_mappoint(self, point):
        """ Add a MapPoint to the stored structure

        :param point: The new map point
        :type point: :py:class:`MapPoint.MapPoint` object

        """
        self.structure.append(point)
