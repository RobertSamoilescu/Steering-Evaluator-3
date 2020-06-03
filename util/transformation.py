from copy import deepcopy
import numpy as np
import cv2



class Transformation(object):
    def __init__(self, K, M):
        """
        :param K: camera intrinsic matrix
        :param M: camera extrinsic matrix
        """
        self.K = deepcopy(K)
        self.M = deepcopy(M)

    def translate_image(self, image, distance):
        """
        :param image: image to transform
        :param distance: positive value for right translation, negative value for left translation
        :return: translated image
        """
        # translation matrix
        T = np.array([
            [1, 0, distance],
            [0, 1, 0],
            [0, 0, 1]
        ])

        # compute projection & homography matrix
        P = np.delete(np.matmul(self.K, self.M), 1, 1)
        H = np.matmul(P, np.matmul(T, np.linalg.inv(P)))

        translated_image = cv2.warpPerspective(image, H, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
        return translated_image

    def rotate_image(self, image, angle):
        """
        :param image: image to transform
        :param angle: positive value (radians) for CW rotation, negative values (radians) for CCW rotation
        :return: rotated image
        """
        # rotation matrix
        R = np.array([
            [np.cos(angle), np.sin(angle), 0],
            [-np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])

        # compute projection & homography matrix
        P = np.delete(np.matmul(self.K, self.M), 1, 1)
        H = np.matmul(P, np.matmul(R, np.linalg.inv(P)))

        rotated_image = cv2.warpPerspective(image, H, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
        return rotated_image


class Crop(object):
    @staticmethod
    def crop_center(image, up=0.5, down=0.5, left=0.5, right=0.5):
        """
        :param image: image to crop from
        :param up: percentage from image height, maximum 50%
        :param down: percentage from image height, maximum 50%
        :param left: percentage from image width, maximum 50%
        :param right: percentage from image width, maximum 50%
        :return: copped image
        """

        # make sure that arguments are in limits
        up = np.maximum(0.0, np.minimum(up, 0.5))
        down = np.maximum(0.0, np.minimum(down, 0.5))
        left = np.maximum(0.0, np.minimum(left, 0.5))
        right = np.maximum(0.0, np.minimum(right, 0.5))

        # compute limits
        center = np.array(image.shape) / 2
        up_limit = up * image.shape[0]
        down_limit = down * image.shape[0]
        left_limit = left * image.shape[1]
        right_limit = right * image.shape[1]

        # define crop corners
        upper_left = (
            np.maximum(0, np.int32(center[0] - up_limit)),
            np.maximum(0, np.int32(center[1] - left_limit))
        )

        lower_right = (
            np.minimum(image.shape[0], np.int32(center[0] + down_limit)),
            np.minimum(image.shape[1], np.int32(center[1] + right_limit))
        )

        return image[upper_left[0]:lower_right[0], upper_left[1]:lower_right[1]]


class Convertor(object):
    @staticmethod
    def kmperh2mperh(speed):
        return speed / 3.6
