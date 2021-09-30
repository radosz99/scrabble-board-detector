import cv2
import os
import numpy as np
import copy
import logging
from operator import itemgetter

MIN_CONT_HEIGHT = 53
MAX_CONT_HEIGHT = 70
MIN_CONT_WIDTH = 1
MAX_CONT_WIDTH = 70
EXTRA_FRAME_SIZE = 6
CANNY_THRESHOLD_1 = 200
CANNY_THRESHOLD_2 = 400
MIN_CONT_AMOUNT = 3
MAX_CONT_AMOUNT = 80
FINAL_SQUARE_SIZE = 12
BOARD_SIZE = 15

logging.basicConfig(filename='demo.log', level=logging.DEBUG)

def get_corners_coordinates(contours):  # first width second height
    width_start, width_end, height_start, height_end = 999, 0, 999, 0
    for contour in contours:
        coord_y, coord_x = contour[0][0], contour[0][1]  # width
        if(coord_y < width_start):
            width_start = coord_y
        if(coord_y > width_end):
            width_end = coord_y
        if(coord_x < height_start):
            height_start = coord_x
        if(coord_x > height_end):
            height_end = coord_x
    return (width_start, width_end, height_start, height_end)

def check_if_letter_coordinates(height, width):
    if(height > MIN_CONT_HEIGHT and height < MAX_CONT_HEIGHT and width > MIN_CONT_WIDTH and width < MAX_CONT_WIDTH):
        return True
    else:
        return False

def get_height_coordinates(start_height, end_height, img_height):
    if(start_height < 0):
        end_height = end_height - start_height
        start_height = 0
    if(end_height > img_height):
        start_height = start_height - (end_height - img_height)
        end_height = img_height
    return start_height, end_height

def get_width_coordinates(start_width, end_width, img_width):
    if(start_width < 0):
        end_width = end_width - start_width
        start_width = 0
    if(end_width > img_width):
        start_width = start_width - (end_width - img_width)
        end_width = img_width
    return start_width, end_width

def get_coordinates_of_square_with_letter_inside(corners_coordinates, img_height, img_width):
    height, width = get_contour_dimensions(corners_coordinates)
    edge_length = height if height > width else width  # calculate square edge
    edge_length += EXTRA_FRAME_SIZE  # add some white space around the contour (letter)
    mphs = (edge_length - height) / 2 if (edge_length - height) % 2 == 0 else (edge_length - height + 1) / 2  # to avoid not ints
    mphe = (edge_length - height) / 2 if (edge_length - height) % 2 == 0 else (edge_length - height - 1) / 2
    mpws = (edge_length - width) / 2 if (edge_length - width) % 2 == 0 else (edge_length - width + 1) / 2
    mpwe = (edge_length - width) / 2 if (edge_length - width) % 2 == 0 else (edge_length - width - 1) / 2
    start_height = corners_coordinates[2] - int(mphs)
    end_height = corners_coordinates[3] + int(mphe)
    start_width = corners_coordinates[0] - int(mpws)
    end_width = corners_coordinates[1] + int(mpwe)
    start_height, end_height = get_height_coordinates(start_height, end_height, img_height)
    start_width, end_width = get_width_coordinates(start_width, end_width, img_width)
    return start_height, end_height, start_width, end_width

def check_if_valid_amount_of_contours(amount):
    if(amount > MAX_CONT_AMOUNT or amount < MIN_CONT_AMOUNT):
        return False
    else:
        return True

def get_image_dimensions(img):
    img_height, img_width, _ = img.shape
    return img_height, img_width

def find_contours_in_img(img):
    edged = cv2.Canny(img, CANNY_THRESHOLD_1, CANNY_THRESHOLD_2)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours

def get_contour_dimensions(coordinates):
    height = coordinates[3] - coordinates[2]
    width = coordinates[1] - coordinates[0]
    return height, width

def get_contour_parameters(contour):
    corners_coordinates = get_corners_coordinates(contour)
    height, width = get_contour_dimensions(corners_coordinates)
    return corners_coordinates, height, width

def get_best_contour_dimensions(contours_list):
    contours_list.sort(key=itemgetter(2),reverse=True)
    return contours_list[0][0], contours_list[0][1], contours_list[0][2]

def get_valid_contours(contours):
    probably_valid_contours = []
    for contour in contours:
        corners_coordinates, height, width = get_contour_parameters(contour)
        if(check_if_letter_coordinates(height, width)):
            probably_valid_contours.append((corners_coordinates, height, width))
    return probably_valid_contours

def put_letter_in_square(corners_coordinates, img):
    img_height, img_width = get_image_dimensions(img) 
    start_height, end_height, start_width, end_width = get_coordinates_of_square_with_letter_inside(corners_coordinates, img_height, img_width)
    cropped_image = img[start_height:end_height, start_width:end_width]  # crop to square
    cropped_image = cv2.resize(cropped_image, dsize=(FINAL_SQUARE_SIZE, FINAL_SQUARE_SIZE), interpolation=cv2.INTER_CUBIC)  # crop to square 12x12
    return cropped_image

def clear_cells(cells_directory, destination_directory):
    images_to_check = os.listdir(cells_directory)
    logging.info(f"Znaleziono {len(images_to_check)} plików z potencjalnymi literami - {images_to_check}")
    for image_name in images_to_check:
        if(os.path.isdir(f"{cells_directory}/{image_name}")):
            continue
        img = cv2.imread(f"{cells_directory}/{image_name}")
        img_height, img_width = get_image_dimensions(img) 
        contours = find_contours_in_img(img)
        logging.info(f"Konturów łącznie w {image_name} - {len(contours)}")
        if(not check_if_valid_amount_of_contours(len(contours))):
            if(len(contours)==1):
                corners_coordinates, height, width = get_contour_parameters(contours[0])
                if(height > int(3/4*img_height) and width > int(3/4*img_width)):  # contour as frame, not possible to detect contours inside
                    cropped_image = img[0:img_height - 5, 0:img_width - 5]  # remove frame
                    contours = find_contours_in_img(cropped_image)
                else:
                    continue
            else:
                continue
                
        probably_valid_contours = get_valid_contours(contours)
        logging.info(f"Walidne koordynaty - {probably_valid_contours}")
        if(probably_valid_contours):
            corners_coordinates, height, width = get_best_contour_dimensions(probably_valid_contours)
            logging.info(f"Wybrane koordynaty - {corners_coordinates}, h: {height}, w: {width}")
            cropped_image = put_letter_in_square(corners_coordinates, img)
            cv2.imwrite(f"{destination_directory}/{image_name}", cropped_image)

def check_if_cell_used(cell_img):
    copy_img = copy.deepcopy(cell_img)
    avg = np.average(copy_img[20])  # random row
    if(avg > 70):  # black - 0, white - 255 ,try to avoid most black cause not used
        return True
    else:
        return False

def divide_board_in_cells(board_filename, destination):
    img = cv2.imread(board_filename)

    height, width, _ = img.shape  # image size
    min_dim = height if height < width else width  # get smaller edge
    min_dim = min_dim - min_dim % BOARD_SIZE  # crop to 15-multiple
    mini = int(min_dim / BOARD_SIZE)  # get cell pixel size
    
    img = cv2.resize(img, dsize=(min_dim, min_dim), interpolation=cv2.INTER_CUBIC)  # crop to square
    
    for multiplier_x in range(BOARD_SIZE):
        for multiplier_y in range(BOARD_SIZE):
            crop_img = img[multiplier_x * mini:(multiplier_x + 1) * mini, multiplier_y * mini :(multiplier_y + 1) * mini].copy()
            if(check_if_cell_used(crop_img)):
                cv2.imwrite(f"{destination}/{multiplier_x}_{multiplier_y}.png", crop_img)

def show_contours(img_path):
    img = cv2.imread(img_path)
    edged = cv2.Canny(img, CANNY_THRESHOLD_1, CANNY_THRESHOLD_2)
    contours, _ = cv2.findContours(edged, 
    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # contours = contours[0:1]

    cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
    cv2.imshow('Contours', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()