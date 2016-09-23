import sys
import os
import cv2

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

           Path of the image dataset

        .. data:: counter

           Number of images in the dataset

        .. data:: image_id

           image_2 identifier, used to track the latest image loaded

        .. data:: cv_window_name

           The name of the window used to display the images

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
        if self.image_id < 9 or self.image_id == 0:
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

    def crop_image(self, start, size, img):
        """ Get a Region Of Interest (ROI).

        Since the images are stored as Numpy arrays we can crop them using the
        Numpy slicing methods.

        :param start: Pixel coordinates of the starting position of the ROI
        :param size: Height and width of the ROI (in pixels)
        :param img: Input image
        :type start: Numpy array 1x2
        :type size: Numpy array 1x2
        :type img: Numpy ndarray
        :returns: The ROI
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
        roi = img[start[1, 0]:start[1, 0] + size[1, 0],
                  start[0, 0]: start[0, 0] + size[0, 0]]
        return roi

    def cleanup(self):
        """ Destroy all windows before exiting the application

        """
        print "Bye..."
        cv2.destroyAllWindows()


def main(args):
    kitti = Dataset('/home/cesar/Documentos/Computer_Vision/01/image_0')
    print kitti.count_images(kitti.path)
    kitti.read_image()
    kitti.show_image()
    kitti.read_image()
    kitti.show_image(1)
    cv2.waitKey(0)

    kitti.cleanup()


if __name__ == '__main__':
    main(sys.argv)
