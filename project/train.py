import training_utils
import os
import letter_detector
import rack_detector

CLASSIFIER_PATH = 'classifiers/final_clf.sav'
IMG_FILE_NAME = 'resources/test.jpg'
TRAINING_WORKSPACE_DIR = 'training'

def collect_samples_from_boards(dir_path):
    dirs_to_check = os.listdir(dir_path)
    for letter in dirs_to_check:
        directory = f"{dir_path}/{letter}"
        if(os.path.exists(f"{directory}/samples")):
            continue
        training_utils.get_boards_from_images(directory, f"{directory}/cropped_boards")
        training_utils.divide_boards_in_cells(f"{directory}/cropped_boards")
        training_utils.leave_only_cells_probably_with_letter(f"{directory}/cropped_boards")
        training_utils.collect_letters_to_one_directory(f"{directory}/samples", f"{directory}/cropped_boards")

def recognize_letters_from_image():
    training_utils.resize_image(2, 'resources/tests/test3.jpg', IMG_FILE_NAME)  # for faster calculations
    board = letter_detector.recognize_letters_from_image(img_path=IMG_FILE_NAME, clf_file_name=CLASSIFIER_PATH)
    print()
    print(board)

# training
# collect_samples_from_boards(TRAINING_WORKSPACE_DIR)
clf = training_utils.get_trained_classifier(TRAINING_WORKSPACE_DIR)
training_utils.save_classifier_to_file(clf, CLASSIFIER_PATH)

# recognizing
# recognize_letters_from_image()

# rack_detector.find_rack_contours('resources/tests/rack2.jpg', CLASSIFIER_PATH)
# letter_detector.save_and_show_contours('resources/tests/rack2.jpg')

