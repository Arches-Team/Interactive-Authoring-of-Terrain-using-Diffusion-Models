
import os

import numpy as np
from PIL import Image

from .constants import GRAD_DOF

PI_OVER_2 = np.pi/2

#################################################################################
#                                                                               #
#                           FORMATS AND CONVERSIONS                             #
#                                                                               #
#-------------------------------------------------------------------------------#
#                                                                               #
#            (1)                  (2)                               (3)         #
# Elevation <---> Gradient Field <---> Scaled Gradient Field (SGF) <---> Image  #
# 2d-array        (dh_dx, dh_dy)               3d-array                   png   #
#                                                                               #
#################################################################################


def elevation_to_gradient(orig_data, xres=1, yres=1):
    t_x_data = np.roll(orig_data, -GRAD_DOF, axis=1)
    t_y_data = np.roll(orig_data, -GRAD_DOF, axis=0)
    dh_dx = (t_x_data - orig_data) / (GRAD_DOF*xres)
    dh_dy = (t_y_data - orig_data) / (GRAD_DOF*yres)

    return dh_dx[:-GRAD_DOF, :-GRAD_DOF], dh_dy[:-GRAD_DOF, :-GRAD_DOF]


def gradient_to_elevation(grad, xres=1, yres=1):
    # TODO test roll vs hstack and slice

    dh_dx, dh_dy = grad

    t_x_data = dh_dx * (GRAD_DOF*xres)
    t_y_data = dh_dy * (GRAD_DOF*yres)

    # A
    t_x_data = np.cumsum(t_x_data, axis=1)
    t_x_data = np.roll(t_x_data, GRAD_DOF, axis=1)
    t_x_data[:, 0] = 0

    # B
    t_y_data = np.cumsum(t_y_data, axis=0)
    t_y_data = np.roll(t_y_data, GRAD_DOF, axis=0)
    t_y_data[0, :] = 0

    # https://stackoverflow.com/questions/11971089/adding-a-vector-to-matrix-rows-in-numpy
    q1 = t_x_data + t_y_data[:, 0, np.newaxis]
    q2 = t_x_data[0] + t_y_data

    return (q1 + q2)/2


def gradient_to_elevation_old(grad, xres=1, yres=1):
    dh_dx, dh_dy = grad
    w, h = dh_dx.shape

    # Ignore last row and column
    c_shape = (w+1, h+1)
    final = np.zeros(c_shape)

    for row, col in np.ndindex(c_shape):
        # print(row, col)
        if row == 0 and col == 0:
            continue  # Keep as 0

        elif row == w and col == h:  # never happens if c_shape = (w, h)
            # Calculate average of 2 neighbours
            final[row, col] = (final[row - GRAD_DOF, col] +
                               final[row, col - GRAD_DOF])/2

        elif col == 0 or row == w:
            final[row, col] = final[row-GRAD_DOF, col] + \
                yres * dh_dy[row-GRAD_DOF, col]

        elif row == 0 or col == h:
            final[row, col] = final[row, col-GRAD_DOF] + \
                xres * dh_dx[row, col-GRAD_DOF]

        else:
            a = final[row, col-GRAD_DOF] + xres * dh_dx[row, col-GRAD_DOF]
            b = final[row-GRAD_DOF, col] + yres * dh_dy[row-GRAD_DOF, col]

            final[row, col] = (a + b)/2

    return final


def gradient_to_SGF(dh_dx, dh_dy):  # TODO change to gradient = dh_dx, dh_dy
    # Store angle in range (-1, 1) instead of raw gradients!
    # Fixes vanishing point and provides more detail in images
    dx = np.arctan(dh_dx) / PI_OVER_2
    dy = np.arctan(dh_dy) / PI_OVER_2
    return np.dstack((dx, dy))


def SGF_to_gradient(sgf):
    dh_dx = np.tan(sgf[:, :, 0] * PI_OVER_2)
    dh_dy = np.tan(sgf[:, :, 1] * PI_OVER_2)

    return dh_dx, dh_dy


def SGF_to_image(final_deriv):
    # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#png
    # First channel for dh/dx
    # Second channel (alpha) for dh/dy

    # (-1,1) -> (0,1)
    final_deriv = (1 + final_deriv)/2

    # Normalise range (0,1) to integers in range (0,255)
    final_deriv = 255 * final_deriv
    final_deriv = np.rint(final_deriv).clip(0, 255).astype(np.uint8)

    return Image.fromarray(final_deriv, 'LA')


def image_to_SGF(img):
    data = np.array(img)
    data = data / 255  # (0,1)
    data = 2 * data - 1  # (-1,1)
    return data


# Helper functions to combine
def elevation_to_SGF(orig_data, xres, yres):
    return gradient_to_SGF(*elevation_to_gradient(orig_data, xres, yres))


def SGF_to_elevation(sgf, xres, yres):
    return gradient_to_elevation(SGF_to_gradient(sgf), xres, yres)

