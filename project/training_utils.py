import cv2
import numpy as np
import os
import logging
from sklearn import svm
import board_detector
import letter_detector
from shutil import copyfile
from sklearn.model_selection import train_test_split
import math
import pickle
from PIL import Image

BOARD_SIZE = 15
TEST_SIZE = 0.5

logging.basicConfig(filename='demo.log', level=logging.DEBUG)

def get_trained_classifier(dir_path):
    target, samples = prepare_data_for_training(dir_path)
    data = samples.reshape((len(samples), -1))  # convert from 12x12 to 1x144
    clf = svm.SVC(gamma=0.001, probability=True)
    X_train, _, y_train, _ = train_test_split(data, target, test_size=TEST_SIZE, shuffle=True)
    clf.fit(X_train, y_train)
    return clf

def convert_image_to_4_bit_array(img_path):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    with np.nditer(image, op_flags=['readwrite']) as it:
        for x in it:
            x[...] = 16 - math.ceil(x / 16)  # 4bit grayscale
    return image

def get_letter_data_from_directory(dir_path):
    files_to_check = os.listdir(dir_path)
    return [convert_image_to_4_bit_array(f"{dir_path}/{file_name}") for file_name in files_to_check], [file_name for file_name in files_to_check]

def prepare_data_for_training(dir_path):
    target, samples = [], []
    dirs_to_check = os.listdir(dir_path)
    for letter in dirs_to_check:
        samples_from_directory, _ = get_letter_data_from_directory(f"{dir_path}/{letter}/samples")
        samples.extend(samples_from_directory)
        target.extend([letter] * len(samples_from_directory))
    samples = np.array(samples)
    return target, samples

def resize_image(divider, img_file_name, new_file_name):
    image = Image.open(img_file_name)
    height, width = image.size
    resized_image = image.resize((int(height/divider),int(width/divider)))
    resized_image.save(new_file_name)

def save_classifier_to_file(clf, clf_file_name):
    pickle.dump(clf, open(clf_file_name, 'wb'))

def create_directory_if_not_exists(directory_path):
    if(not os.path.exists(directory_path)):
        os.makedirs(directory_path)

def create_directories_if_not_exists(list_of_dirs):
    for dir in list_of_dirs:
        create_directory_if_not_exists(dir)

def get_boards_from_images(dir_path, dest_dir_path):
    create_directory_if_not_exists(dest_dir_path)
    files_to_check = os.listdir(dir_path)
    for file_name in files_to_check:
        if(os.path.isfile(f"{dir_path}/{file_name}")):
            logging.info(f"{dir_path}/{file_name}")
            board = board_detector.get_board_from_image(ref_img_path="resources/board_empty.jpg", board_img_path=f"{dir_path}/{file_name}")
            cv2.imwrite(f"{dest_dir_path}/{file_name}_cropped.png", board)

def divide_boards_in_cells(boards_directory):
    files_to_check = os.listdir(boards_directory)
    for file_name in files_to_check:
        if(os.path.isfile(f"{boards_directory}/{file_name}")):
            logging.info(f"{boards_directory}/{file_name}")
            create_directory_if_not_exists(f"{boards_directory}/{file_name}_cells")
            letter_detector.divide_board_in_cells(f"{boards_directory}/{file_name}", f"{boards_directory}/{file_name}_cells") 

def leave_only_cells_probably_with_letter(cells_directory):
    files_to_check = os.listdir(cells_directory)
    for file_name in files_to_check:
        if(os.path.isdir(f"{cells_directory}/{file_name}")):
            logging.info(f"{cells_directory}/{file_name}")
            create_directory_if_not_exists(f"{cells_directory}/{file_name}/cleared")
            letter_detector.leave_only_cells_with_contours_similar_to_letter(f"{cells_directory}/{file_name}", f"{cells_directory}/{file_name}/cleared")

def collect_letters_to_one_directory(destination_directory, source_directory):
    create_directory_if_not_exists(destination_directory)
    files_to_check = os.listdir(source_directory)
    for file_name in files_to_check:
        if(os.path.isdir(f"{source_directory}/{file_name}")):
            files_to_check_2 = os.listdir(f"{source_directory}/{file_name}/cleared")
            for file_name_2 in files_to_check_2:
                if(os.path.isfile(f"{source_directory}/{file_name}/cleared/{file_name_2}")):
                    copyfile(f"{source_directory}/{file_name}/cleared/{file_name_2}", f"{destination_directory}/{file_name}_{file_name_2}")
