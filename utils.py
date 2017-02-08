import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


"""
..  codeauthor:: Cesar Gonzalez Gonzalez
:file utils.py
"""


def np_to_df(array, headers):
    """ Writes a numpy array to a pandas DataFrame

    :param array: The array to be transformed
    :param headers: Columns name.
    :type array: Numpy array
    :type headers: List

    :returns: None
    """
    return pd.DataFrame(array, columns=headers)


def concat_df(frames, axis=1):
    """ Concatenate Dataframes

    :param frames: List of DataFrames to be concatenated
    :param axis: The axis to concatenate along (1 means horizontal)
    :type frames: List
    :type axis: Integer
    :returns: The concatenated DataFrame
    :rtype: Pandas DataFrame
    """
    return pd.concat(frames, axis=axis)


def draw_epilines(vo):
    """ Draw epilines in both images

    :param vo: Instance of the :py:class:`VisualOdometry.VisualOdometry` object.
    :type vo: :py:class:`VisualOdometry.VisualOdometry` object.
    """
    img1, img2 = vo.draw_epilines(vo.kitti.image_1.copy(),
                                  vo.kitti.image_2.copy(),
                                  vo.find_epilines(vo.matcher.kp_list_to_np(vo.matcher.good_kp2)),
                                  vo.matcher.kp_list_to_np(vo.matcher.good_kp1),
                                  vo.matcher.kp_list_to_np(vo.matcher.good_kp2))
    img3, img4 = vo.draw_epilines(vo.kitti.image_2.copy(),
                                  vo.kitti.image_1.copy(),
                                  vo.find_epilines(vo.matcher.kp_list_to_np(vo.matcher.good_kp1)),
                                  vo.matcher.kp_list_to_np(vo.matcher.good_kp2),
                                  vo.matcher.kp_list_to_np(vo.matcher.good_kp1))
    plt.subplot(121), plt.imshow(img1)
    plt.subplot(122), plt.imshow(img3)
    plt.show()


def plot_matches(vo):
    """ Plot the correspondences found in two images

    :param vo: Instance of the :py:class:`VisualOdometry.VisualOdometry` object.
    :type vo: :py:class:`VisualOdometry.VisualOdometry` object.
    """
    img = vo.matcher.draw_matches(vo.kitti.image_2.copy(),
                                  vo.matcher.good_matches,
                                  color='red')
    plt.imshow(img)


def plot_scene(point_cloud):
    """ Plot a 3D point cloud (the scene)

    :param point_cloud: 3D point cloud, :math:`(X, Y, Z)`
    :type point_cloud: Numpy (n, 3) ndarray
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(point_cloud[:, 0], point_cloud[:, 1],
            point_cloud[:, 2],
            c='r',
            marker='o',
            linestyle="")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
