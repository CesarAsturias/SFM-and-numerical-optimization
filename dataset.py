import sys
import os
import cv2
import pandas as pd
import numpy as np

"""
.. codeauthor:: Cesar Gonzalez Gonzalez <cesargg.luarca@gmail.com>
 :file dataset.py

 """


class Dataset(object):
    """ Basic class to manage the images

    This class represents the Kitti dataset.

    **Attributes**:

        .. data:: image_1

           The first image loaded. Numpy ndarray

        .. data:: image_2

           The second image loaded. Numpy ndarray

        .. data:: image_size

           Size of the images loaded

        .. data:: image_width

           Width of the images

        .. data:: image_height

           Height of the images

        .. data:: path

           Path to the image dataset

        .. data:: counter

           Number of images in the dataset

        .. data:: image_id

           image_2 identifier, used to track the latest image loaded

        .. data:: cv_window_name

           The name of the window used to display the images

        .. data:: ground_truth

           Ground truth coordinates for the image sequence. Numpy nx12 ndarray.

    **Constructor**:

        The constructor takes as argument the path (string) to the image dataset.

    **Members**:

    """

    def __init__(self, path):
        # Initialize the attributes
        self.image_size = None
        self.image_width = None
        self.image_height = None
        self.path = str(path)
        self.counter = 0
        self.image_1 = None
        self.image_2 = None
        self.image_id = 0
        self.cv_window_name = "Kitti image"

        # Count the number of images in the path
        self.counter = self.count_images(self.path)

    def read_image(self):
        """ Reads a new image from the dataset path

        The Kitti_ dataset images
        are named like *xxxxxx.png*, so we should create the correct name
        based on the **image_id** attribute. Basically, we add the number of
        zeros neccesary for the image name to fullfill the kitty standard image
        names.

        .. _Kitti: http://www.cvlibs.net/datasets/kitti/

        """
        if self.image_id <= 9 or self.image_id == 0:
            number_image = '00000' + str(self.image_id)
        elif self.image_id < 100 and self.image_id > 9:
            number_image = '0000' + str(self.image_id)
        elif self.image_id < 1000 and self.image_id >= 100:
            number_image = '000' + str(self.image_id)
        elif self.image_id < 10000 and self.image_id >= 1000:
            number_image = '00' + str(self.image_id)
        elif self.image_id < 100000 and self.image_id >= 10000:
            number_image = '0' + str(self.image_id)
        else:
            number_image = str(self.image_id)

        file_name = self.path + '/' + number_image + '.png'
        # print("New image: {}".format(file_name))

        # Now, read the image
        # If we are loading the first image of the dataset we will store it
        # in the **image_2** attribute. **image_1** will remain unaltered
        # (None). Otherwise, we copy **image_2** to **image_1** using the *copy*
        # method of numpy.

        if self.image_id != 0:
            self.image_1 = self.image_2.copy()

        self.image_2 = cv2.imread(file_name)

        self.image_height = self.image_2.shape[0]

        self.image_width = self.image_2.shape[1]

        self.image_id += 1

    def show_image(self, image=0):
        """ Plot an image

        :param image: This parameter is used to determine which image is going
                      to be displayed:

                        a. 0 --> Show the last image

                        b. 1 --> Show the first image

        :type image: Integer

        """
        if image == 0:
            cv2.imshow(self.cv_window_name, self.image_2)
        else:
            cv2.imshow(self.cv_window_name, self.image_1)
        self.keystroke = cv2.waitKey(0) & 0XFF
        if self.keystroke == 27:
            cv2.DestroyAllWindows()
        elif self.keystroke == ord('s'):
            cv2.imwrite('Image.png', self.image_2)
            cv2.DestroyAllWindows()

    def copy_image(self):
        """ Copy the new image to the previous image
        """
        self.image_1 = self.image_2.copy()

    def count_images(self, path):
        """ Count the number of images in the specified path.
        :param path: The dataset path.
        :type path: String
        :returns: The number of images of the dataset
        :rtype: Integer

        """

        count = len([name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))])
        return count

    def crop_image(self, start, size, img, center=False):
        """ Get a Region Of Interest (ROI).

        Since the images are stored as Numpy arrays we can crop them using the
        Numpy slicing methods.

        :param start: Pixel coordinates of the starting position of the ROI,
                      :math:`(p_x, p_y)`
        :param size: Height and width of the ROI (in pixels),
                     :math:`(w, h)`
        :param img: Input image
        :param center: If True, then use the start parameter as the center of
                       the ROI.
        :type start: Numpy array 1x2
        :type size: Numpy array 1x2
        :type img: Numpy ndarray
        :type center: Boolean
        :returns: The ROI or None if the ROI lies outside the image borders.
        :rtype: Numpy ndarray

        :Example:

            >>> from dataset import dataset
            >>> import numpy as np
            >>> path = '/home/cesar/Documentos/Computer_Vision/01/image_0'
            >>> start = np.array([0, 0])
            >>> size = np.array([10, 10])
            >>> kitti = Dataset(path)
            >>> kitti.read_image()
            >>> kitti.crop_image(start, size, kitti.image_2)

        """
        img = img.copy()
        if center:
            if (start[0] - size[0] / 2 < 0 or
                start[0] + size[0] / 2 > self.image_2.shape[1]):
                print ("X coordinates not in image")
                return None
            elif (start[1] - size[1] / 2 < 0 or
                  start[1] + size[1] / 2 > self.image_2.shape[0]):
                print ("Y coordinates not in image")
                return None
            else:
                val = int(round(size[0] / 2))
                x_init = int(start[0] - val)
                x_end = int(start[0] + val)
                val = int(round(size[1] / 2))
                y_init = int(start[1] - val)
                y_end = int(start[1] + val)
                return img[y_init:y_end,x_init:x_end]
                
        else:
            roi = img[start[0]:start[0] + size[0],
                      start[1]: start[1] + size[1]]
            return roi

    def read_ground_truth(self, filename):
        """ Reads the ground truth poses :math:`T` of the image sequence.

        The method reads a txt file which has the following shape:

        .. math::

            T_i = (R_{11},R_{12},R_{13},t_{x},R_{21},R_{22},R_{23},t_{y},
                     R_{31},R_{32},R_{33},t_{z})

        where :math:`T_i` is the i-th row of the file.

        :param filename: The **absolute** path to the txt file.
        :type filename: String
        :returns: The ground truth poses as a numpy ndarray of shape
                  (n, 3, 4), where :math:`n` is the number of rows (poses) and
                  for each :math:`n` there is a 3x4 matrix:

                  .. math::

                        T_i = [R_i|\\mathbf{t}]

        :rtype: Numpy ndarray

        Example::

            >>> from dataset import Dataset
            >>> from mpl_toolkits.mplot3d import Axes3D
            >>> import matplotlib.pyplot as plt
            >>> kitti = Dataset('path_to_images')
            >>> poses = kitti.read_ground_truth('path_to_poses')
            >>> # Extract X, Y and Z coordinates
            >>> X = poses[:, 0, 3]
            >>> Y = poses[:, 1, 3]
            >>> Z = poses[:, 2, 3]
            >>> # Plot the 3D trajectory
            >>> fig = plt.figure()
            >>> ax = fig.add_subplot(111, projection='3d')
            >>> ax.scatter(X, Y, Z)
            >>> plt.xlabel('X')
            >>> plt.ylabel('Y')
            >>> plt.show()
            >>> # Plot 2D trajectory
            >>> fig_2d = plt.figure()
            >>> plt.plot(X, Z)
            >>> plt.show()

        The above example will plot the following images:

        .. image:: ../Images/traj_3d_truth.png

        .. image:: ../Images/traj_2d_truth.png


        """
        names = ['R11', 'R12', 'R13', 'X',
                 'R21', 'R22', 'R23', 'Y',
                 'R31', 'R32', 'R33', 'Z']
        poses = pd.read_csv(filename, header=None, names=names,
                            delim_whitespace=True)
        poses_array = poses.as_matrix()
        lst = []
        for i in range(len(poses_array)):
            lst.append(poses_array[i].reshape((3, 4)))
        poses_array = np.asarray(lst)
        return poses_array

    def cleanup(self):
        """ Destroy all windows before exiting the application

        """
        print ("Bye...")
        cv2.destroyAllWindows()


def main(args):
    kitti = Dataset('/home/cesar/Documentos/Computer_Vision/01/image_0')
    print (kitti.count_images(kitti.path))
    kitti.read_image()
    kitti.show_image()
    kitti.read_image()
    kitti.show_image(1)
    cv2.waitKey(0)

    kitti.cleanup()


if __name__ == '__main__':
    main(sys.argv)
