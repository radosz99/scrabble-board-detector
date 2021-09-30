import cv2
import numpy as np
import os
import imutils
import xml.etree.ElementTree as ET
import re
from numpngw import write_png
import math

middle = 0

def perspective_transform(image, corners):
    def order_corner_points(corners):
        corners = [(corner[0][0], corner[0][1]) for corner in corners]
        top_r, top_l, bottom_l, bottom_r = corners[0], corners[1], corners[2], corners[3]
        return (top_l, top_r, bottom_r, bottom_l)

    ordered_corners = order_corner_points(corners)
    top_l, top_r, bottom_r, bottom_l = ordered_corners

    width_A = np.sqrt(((bottom_r[0] - bottom_l[0]) ** 2) + ((bottom_r[1] - bottom_l[1]) ** 2))
    width_B = np.sqrt(((top_r[0] - top_l[0]) ** 2) + ((top_r[1] - top_l[1]) ** 2))
    width = max(int(width_A), int(width_B))

    height_A = np.sqrt(((top_r[0] - bottom_r[0]) ** 2) + ((top_r[1] - bottom_r[1]) ** 2))
    height_B = np.sqrt(((top_l[0] - bottom_l[0]) ** 2) + ((top_l[1] - bottom_l[1]) ** 2))
    height = max(int(height_A), int(height_B))
    dimensions = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], 
                    [0, height - 1]], dtype = "float32")

    ordered_corners = np.array(ordered_corners, dtype="float32")
    matrix = cv2.getPerspectiveTransform(ordered_corners, dimensions)
    return cv2.warpPerspective(image, matrix, (width, height))

def convert_image_to_grayscale(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]


def convert_images_from_directory_to_grayscale(dir_path):
    images_to_check = os.listdir(dir_path)
    for image_name in images_to_check:
        image_path = f"{dir_path}/{image_name}"
        threshold_img = convert_image_to_grayscale(image_path)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
        threshold_img = cv2.morphologyEx(threshold_img, cv2.MORPH_OPEN, kernel)
        cv2.imwrite(image_path, threshold_img)


def retrieve_board_from_image(image_path):
    image = cv2.imread(image_path)
    original = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 3)
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,3)

    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.015 * peri, True)
        return perspective_transform(original, approx)

def retrieve_board_from_images_from_directory(dir_path, dest_dir_path):
    if(not os.path.exists(dest_dir_path)):
        os.makedirs(dest_dir_path)
    images_to_check = os.listdir(dir_path)
    for image_name in images_to_check:
        if(os.path.isfile(f"{dir_path}/{image_name}")):
            print(f"{dir_path}/{image_name}")
            board = retrieve_board_from_image(f"{dir_path}/{image_name}")
            cv2.imwrite(f"{dest_dir_path}/{image_name}_cropped.png", board)

def get_letters(image_path):
    pass

directory = 'training/e'
destination_directory = 'training/e/cropped_boards'
# retrieve_board_from_images_from_directory(directory, destination_directory)
# convert_images_from_directory_to_grayscale(destination_directory)


 