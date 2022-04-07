"""
image_processing.py
    Main File for Image Processing Sub routines

    @author Nicholas Nordstrom
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt


def visualize_connected_components(image, min_pixels=0, max_pixels=-1, show=False, print_info=False):
    """
    @author Nicholas Nordstrom
    Method to perform connected component analysis on an image and visualize the results
    :param image: image to perform connected components on (assumes to be binary)
    :param min_pixels: minimum number of pixels to be considered a component
    :param max_pixels: maximum number of pixels to be considered a component
    :param show: option to show images of each component
    :param print_info: option to print info about image, labels and component
    :return: list of masks excluding each component that meets requirements
    """

    # get connected components from cv2
    num_labels, labels = cv2.connectedComponents(np.uint8(image), connectivity=8)

    # print important info
    if print_info:
        print('image shape:', np.shape(image))
        print('0 element of image:', image[0][0])
        print('label shape:', np.shape(labels))
        print('num labels:', num_labels)
        print('labels:', labels[labels != 0])
        print('max in labels:', np.max(labels))

    # list to hold connected component masks
    cc_masks = []

    for i in range(num_labels): # for each component found...

        # calculate number of pixels
        num_pixels = np.sum(labels == i)

        # ignore this component if it does not meet requirements
        if num_pixels > max_pixels != -1 or num_pixels < min_pixels:
            continue

        # print info about this specific component
        if print_info:
            print("component {} has {} pixels".format(i, num_pixels))

        # create an image showing just THIS component
        image_copy = image.copy()
        image_copy[labels != i] = 0
        image_copy[labels == i] = 1

        # append this connected component mask to the list
        cc_masks.append(tuple([labels != i]))

        # show the image using matplotlib
        if show:
            plt.imshow(image_copy, cmap='gray')
            plt.show()

    # return list of connected components
    return cc_masks
