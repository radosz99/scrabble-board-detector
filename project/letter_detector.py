import cv2
import os
import numpy as np
import copy
import logging
from operator import itemgetter
import pickle
import math

from numpy.lib.function_base import average
import training_utils
import board_detector

MIN_CONT_HEIGHT = 53
MAX_CONT_HEIGHT = 70
MIN_CONT_WIDTH = 1
MAX_CONT_WIDTH = 80
EXTRA_FRAME_SIZE = 6
CANNY_THRESHOLD_1 = 200
CANNY_THRESHOLD_2 = 400
MIN_CONT_AMOUNT = 3
MAX_CONT_AMOUNT = 80
FINAL_SQUARE_SIZE = 12
BOARD_SIZE = 15
FRAME_PROPORTION_MIN = 0.75
FRAME_SIZE = 5
MIN_LETTER_PROBA = 0.12

logging.basicConfig(filename='demo.log', level=logging.DEBUG)

def recognize_letters_from_image(img_path, clf_file_name, output_directory='output'):
    prepare_output_workspace_for_detecting(output_directory)
    clf = get_classifier_from_file(clf_file_name)
    save_detected_board_in_file(img_path, f"{output_directory}/cropped.png")
    divide_board_in_cells(f"{output_directory}/cropped.png", f"{output_directory}/cells")
    leave_only_cells_with_contours_similar_to_letter(f"{output_directory}/cells", f"{output_directory}/cleared")

    predicted, prob_predicted, letters_file_names = predict_letters(clf, f"{output_directory}/cleared")
    board_array = get_filled_board_based_on_predicted_letters(predicted, prob_predicted, letters_file_names, clf.classes_)
    return board_array


def get_filled_board_based_on_predicted_letters(predicted, prob_predicted, letters_file_names, classes):
    board_array = init_array_for_board()
    for i, letter_file_name in enumerate(letters_file_names):
        coord_x, coord_y = get_coordinates_on_board_from_file_name(letter_file_name)
        prob_with_classes = list(zip(classes, prob_predicted[i]))
        prob_with_classes.sort(key=lambda tup: tup[1], reverse=True)
        logging.info(f"{coord_x}_{coord_y}, prob - {prob_with_classes}, pred - {predicted[i]}")
        put_letter_on_board_if_valid(max(prob_predicted[i]), predicted[i], coord_x, coord_y, board_array)
    return board_array


def init_array_for_board():
    return np.full((BOARD_SIZE, BOARD_SIZE), ' ')


def prepare_output_workspace_for_detecting(output_directory):
    training_utils.create_directories_if_not_exists([output_directory, f"{output_directory}/cells", f"{output_directory}/cleared"])


def put_letter_on_board_if_valid(proba, predicted_letter, coord_x, coord_y, board_array):
    if(proba >= MIN_LETTER_PROBA):
            board_array[coord_x, coord_y] = predicted_letter


def get_coordinates_on_board_from_file_name(file_name):
    coords = file_name[:-4].split('_')  # 11_0.png -> [11, 0]
    return int(coords[0]), int(coords[1])


def predict_letters(clf, letter_data_dir):
    letters, letters_file_names = training_utils.get_letter_data_from_directory(letter_data_dir)
    letters = np.array(letters)
    data = letters.reshape((len(letters), -1))

    prob_predicted = [[round(prob, 3) for prob in letter] for letter in clf.predict_proba(data)]
    predicted = clf.predict(data)
    return predicted, prob_predicted, letters_file_names


def save_detected_board_in_file(img_path, filename):
    cv2.imwrite(filename, board_detector.get_board_from_image(ref_img_path="resources/board_empty.jpg", board_img_path=f"{img_path}"))


def get_classifier_from_file(clf_file_name):
    return pickle.load(open(clf_file_name, 'rb'))


def get_corners_coordinates(contours):  # first width second height
    width_start, width_end, height_start, height_end = 99999, 0, 99999, 0
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
    # edge_length += 30  # add some white space around the contour (letter)
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
    contours_list.sort(key=itemgetter(2),reverse=True)  # sort by width from largest
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
    # cropped_image = cv2.resize(cropped_image, dsize=(100, 100), interpolation=cv2.INTER_CUBIC)  # crop to square 12x12
    return cropped_image


def leave_only_cells_with_contours_similar_to_letter(cells_directory, destination_directory):
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
                _, height, width = get_contour_parameters(contours[0])
                distract_letter_from_image_where_only_one_contour_is_frame(height, width, img_height, img_width, img)
            else:
                continue
                
        probably_valid_contours = get_valid_contours(contours)
        logging.info(f"Walidne koordynaty - {probably_valid_contours}")
        if(probably_valid_contours):
            save_contours_to_file(probably_valid_contours, img, f"{destination_directory}/{image_name}")

def distract_letter_from_image_where_only_one_contour_is_frame(height, width, img_height, img_width, img):
    if(check_if_contour_is_frame(height, width, img_height, img_width)):  # contour as frame, not possible to detect contours inside
        cropped_image = img[0:img_height - FRAME_SIZE, 0:img_width - FRAME_SIZE]  # remove frame
        return find_contours_in_img(cropped_image)
    else:
        return []

def check_if_contour_is_frame(height, width, img_height, img_width):
    return True if height > int(FRAME_PROPORTION_MIN*img_height) and width > int(FRAME_PROPORTION_MIN*img_width) else False

def save_contours_to_file(contours_list, img, file_name):
    corners_coordinates, height, width = get_best_contour_dimensions(contours_list)
    logging.info(f"Wybrane koordynaty - {corners_coordinates}, h: {height}, w: {width}")
    cropped_image = put_letter_in_square(corners_coordinates, img)
    cv2.imwrite(file_name, cropped_image)

def check_if_cell_is_used(cell_img):
    return True if np.average(cell_img[20]) > 70 else False # black - 0, white - 255, trying to avoid most black cause that cells are not used


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
            if(check_if_cell_is_used(crop_img)):
                cv2.imwrite(f"{destination}/{multiplier_x}_{multiplier_y}.png", crop_img)

def save_binarized_image(source, destination):
    img = cv2.imread(source)
    img = board_detector.binarize_image(img)
    cv2.imwrite(destination, img)

def save_and_show_contours(img_path):
    img = cv2.imread(img_path)
    contours = find_contours_in_img(img)
    contours_list = []
    for contour in contours:
        contours_list.append((get_contour_parameters(contour), contour))
    contours_list.sort(key=lambda tup: tup[0][1], reverse=True)
    for contour in contours_list[:20]:
        print(contour[0])
    best_contours = [contour[1] for contour in contours_list]
    cv2.drawContours(img, best_contours[:20], -1, (0, 255, 0), 3)
    cv2.imwrite("img_with_contours.png", img)
    # cv2.imshow('Contours', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()