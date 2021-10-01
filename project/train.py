import training_utils
import os
import letter_detector

CLASSIFIER_PATH = 'classifiers/final_clf.sav'
IMG_FILE_NAME = 'resources/test.jpg'


def collect_samples_from_boards(dir_path):
    dirs_to_check = os.listdir(dir_path)
    for letter in dirs_to_check:
        directory = f"{dir_path}/{letter}"
        training_utils.get_boards_from_images(directory, f"{directory}/cropped_boards")
        training_utils.divide_boards_in_cells(f"{directory}/cropped_boards")
        training_utils.leave_only_cells_probably_with_letter(f"{directory}/cropped_boards")
        training_utils.collect_letters_to_one_directory(f"{directory}/samples", f"{directory}/cropped_boards")

# collect_samples_from_boards('training')
# clf = training_utils.get_trained_classifier('training')
# training_utils.save_classifier_to_file(clf)

# training_utils.resize_image(2, 'xdd.png', 'xd.png')

training_utils.resize_image(2, 'resources/tests/test.jpg', IMG_FILE_NAME)
board = letter_detector.recognize_letters_from_image(img_path=IMG_FILE_NAME, clf_file_name=CLASSIFIER_PATH)
print(board)