import cv2
import numpy as np
import os
import logging
from sklearn import svm
from . import board_detector
from . import letter_detector
from shutil import copyfile
from sklearn.model_selection import train_test_split
import math
import pickle
from PIL import Image

MIN_LETTER_PROBA = 0.35
BOARD_SIZE = 15
TEST_SIZE = 0.5

logging.basicConfig(filename='demo.log', level=logging.DEBUG)

def recognize_letters_from_image(img_path, clf_file_name, output_directory='output'):
    create_directory_if_not_exists(output_directory)
    create_directory_if_not_exists(f"{output_directory}/cells")
    create_directory_if_not_exists(f"{output_directory}/cleared")

    clf = get_classifier_from_file(clf_file_name)
    cv2.imwrite(f"{output_directory}/cropped.png", board_detector.get_board_from_image(ref_img_path="resources/board_empty.jpg", board_img_path=f"{img_path}"))
    letter_detector.divide_board_in_cells(f"{output_directory}/cropped.png", f"{output_directory}/cells")
    letter_detector.clear_cells(f"{output_directory}/cells", f"{output_directory}/cleared")

    letters, file_names = get_letter_data_from_directory(f"{output_directory}/cleared")
    letters = np.array(letters)
    data = letters.reshape((len(letters), -1))

    prob_predicted = [[round(prob, 3) for prob in letter] for letter in clf.predict_proba(data)]
    predicted = clf.predict(data)
    
    board_array = np.full((BOARD_SIZE, BOARD_SIZE), ' ')

    for i, file_name in enumerate(file_names):
        coords = file_name[:-4].split('_')  # 11_0.png -> [11, 0]
        coord_x = int(coords[0])
        coord_y = int(coords[1])
        logging.info(f"{coord_x}_{coord_y}, prob - {list(zip(clf.classes_, prob_predicted[i]))}, pred - {predicted[i]}")
        if(max(prob_predicted[i]) >= MIN_LETTER_PROBA):
            board_array[coord_x, coord_y] = predicted[i]
    return board_array

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

def collect_samples_from_boards(dir_path):
    dirs_to_check = os.listdir(dir_path)
    for letter in dirs_to_check:
        directory = f"{dir_path}/{letter}"
        get_boards_from_images(directory, f"{directory}/cropped_boards")  # retrieve boards
        divide_boards_in_cells(f"{directory}/cropped_boards", f"{directory}/cropped_boards")
        get_letter_from_cells_from_boards(f"{directory}/cropped_boards")
        collect_letters_to_one_directory(f"{directory}/samples", f"{directory}/cropped_boards")

def resize_image(divider, img_file_name):
    image = Image.open(img_file_name)
    height, width = image.size
    resized_image = image.resize((int(height/divider),int(width/divider)))
    resized_image.save(img_file_name)

def save_classifier_to_file(clf_file_name):
    pickle.dump(get_trained_classifier('training'), open(clf_file_name, 'wb'))

def get_classifier_from_file(clf_file_name):
    return pickle.load(open(clf_file_name, 'rb'))

def create_directory_if_not_exists(directory_path):
    if(not os.path.exists(directory_path)):
        os.makedirs(directory_path)

def get_boards_from_images(dir_path, dest_dir_path):
    create_directory_if_not_exists(dest_dir_path)
    files_to_check = os.listdir(dir_path)
    for file_name in files_to_check:
        if(os.path.isfile(f"{dir_path}/{file_name}")):
            logging.info(f"{dir_path}/{file_name}")
            board = board_detector.get_board_from_image(ref_img="resources/board_empty.jpg", board_img=f"{dir_path}/{file_name}")
            cv2.imwrite(f"{dest_dir_path}/{file_name}_cropped.png", board)

def divide_boards_in_cells(boards_directory, destination_directory):
    create_directory_if_not_exists(destination_directory)
    files_to_check = os.listdir(boards_directory)
    for file_name in files_to_check:
        if(os.path.isfile(f"{boards_directory}/{file_name}")):
            logging.info(f"{boards_directory}/{file_name}")
            create_directory_if_not_exists(f"{boards_directory}/{file_name}_cells")
            letter_detector.divide_board_in_cells(f"{boards_directory}/{file_name}", f"{boards_directory}/{file_name}_cells") 

def get_letter_from_cells_from_boards(cells_directory):
    files_to_check = os.listdir(cells_directory)
    for file_name in files_to_check:
        if(os.path.isdir(f"{cells_directory}/{file_name}")):
            logging.info(f"{cells_directory}/{file_name}")
            create_directory_if_not_exists(f"{cells_directory}/{file_name}/cleared")
            letter_detector.clear_cells(f"{cells_directory}/{file_name}", f"{cells_directory}/{file_name}/cleared")

def collect_letters_to_one_directory(destination_directory, source_directory):
    create_directory_if_not_exists(destination_directory)
    files_to_check = os.listdir(source_directory)
    for file_name in files_to_check:
        if(os.path.isdir(f"{source_directory}/{file_name}")):
            files_to_check_2 = os.listdir(f"{source_directory}/{file_name}/cleared")
            for file_name_2 in files_to_check_2:
                if(os.path.isfile(f"{source_directory}/{file_name}/cleared/{file_name_2}")):
                    copyfile(f"{source_directory}/{file_name}/cleared/{file_name_2}", f"{destination_directory}/{file_name}_{file_name_2}")
