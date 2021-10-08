import cv2
import logging
import math
from numpy.lib.function_base import average
import letter_detector
import training_utils

HEIGHT_SIMILARITY_PERCENT = 0.12
WIDTH_HEIGH_MAX_RATIO = 1.15

def check_if_width_is_covering_in_two_contours(first_contour, second_contour):
    start_width_first = first_contour[0]
    end_width_first = first_contour[1]
    start_width_second = second_contour[0]
    end_width_second = second_contour[1]
    if(start_width_first < start_width_second): 
        if(end_width_first < start_width_second): 
            return False
        else: 
            return True
    else:
        if(end_width_second < start_width_first):
            return False
        else:
            return True

def get_first_height_from_list_of_contours(contours):
    if(contours):
        return contours[0][1]
    else:
        return None

def get_indexes_of_lists_in_which_candidates_has_got_similar_height(height, all_candidates_contours):
    indexes_list = []
    for i, candidates_contours in enumerate(all_candidates_contours):
        print(f"Sprawdzam kandydata do podobieństwa w wysokości - {i}")
        height_from_other_contours = get_first_height_from_list_of_contours(candidates_contours)
        print(f"Wysokość konturu - {height}, wysokość konturu z kandydata - {height_from_other_contours}")
        diff = math.fabs(height_from_other_contours - height)
        print(f"Diff = {diff}, max = {HEIGHT_SIMILARITY_PERCENT * height_from_other_contours}")
        if(diff < HEIGHT_SIMILARITY_PERCENT * height):
            print("Wysokość jest podobna, dodajemy indeks")
            indexes_list.append(i)
        else:
            print("Niepodobna wysokość, nie dodajemy")
    return indexes_list

def add_to_list_if_contours_is_not_covering_with_other_contours(contour, all_candidates_contours, candidates_indexes):
    for i, candidates_contours in enumerate(all_candidates_contours):
        not_covering = True
        print(f"Sprawdzam kandydata do pokrywania się w szerokości - {i}")
        if(not i in candidates_indexes):
            continue
        else:
            for candidate_contour in candidates_contours:
                if(check_if_width_is_covering_in_two_contours(contour[0], candidate_contour[0])):
                    print(f"Pokrywa się z {candidate_contour[0]} z listy nr {i}, tworzymy nowy kandydujący zbiór")
                    new_candidates_contours = list()
                    new_candidates_contours.append(contour)
                    all_candidates_contours.append(new_candidates_contours)
                    not_covering = False
                    break
        if(not_covering):
            print("Nie pokrywa się, dodajemy")
            candidates_contours.append(contour)

def check_if_valid_contour_by_width_height_ratio(width, height):
    if((width / height) < WIDTH_HEIGH_MAX_RATIO):
        return True
    else:
        return False

def find_letters_in_rack(contours_to_check):
    letters_contours = []
    for contour in contours_to_check:
        print(f"Obecne zbiory kandydujących konturów - {letters_contours}")
        print(f"Sprawdzam kontur: {contour}")
        if(not check_if_valid_contour_by_width_height_ratio(contour[2], contour[1])):
            print("Ratio szerokość / wysokość się nie zgadza")
            continue
        contours_candidates_indexes = get_indexes_of_lists_in_which_candidates_has_got_similar_height(contour[1], letters_contours)
        print(f"Podobna wysokość do kandydatów - {contours_candidates_indexes}")
        if(not contours_candidates_indexes):
            print("Tworzę nową listę, niepodobna wysokość")
            new_list = list()
            new_list.append(contour)
            letters_contours.append(new_list)
        else:
            add_to_list_if_contours_is_not_covering_with_other_contours(contour, letters_contours, contours_candidates_indexes)
    return letters_contours

def get_height_variance(candidates_contours):
    heights_from_candidates_contours = [contour[0][2] for contour in candidates_contours]
    avg_height = average(heights_from_candidates_contours)
    print(f"Candidates - {candidates_contours}, average - {avg_height}")
    sum_var = 0
    for contour in candidates_contours:
        sum_var += math.pow(contour[0][2] - avg_height, 2)
    return sum_var / len(candidates_contours)
    
def evaluate_contours(all_candidates_contours):
    best_canditates_contours = []
    for candidates_contours in all_candidates_contours:
        if(len(candidates_contours) > len(best_canditates_contours)):
            best_canditates_contours = candidates_contours
    return best_canditates_contours

def get_most_probable_letter_set(all_candidates_contours):
    return evaluate_contours(all_candidates_contours)

def find_rack_contours(img_path, clf_file_name, output_directory='output'):
    training_utils.create_directories_if_not_exists([output_directory, f"{output_directory}/rack"])
    dest_path = "img_with_contours.png"
    letter_detector.save_binarized_image(img_path, dest_path)
    img = cv2.imread(dest_path)
    # img = cv2.imread(img_path)
    contours = letter_detector.find_contours_in_img(img)
    contours_list = []
    for contour in contours:
        contours_list.append(letter_detector.get_contour_parameters(contour))
    contours_list.sort(key=lambda tup: tup[1], reverse=True)
    letters_sets = find_letters_in_rack(contours_list[:30])
    # for letter_set in letters_sets:
    #     print(f"Elementów : {len(letter_set)} - {letter_set}")
    #     print()
    letter_set = get_most_probable_letter_set(letters_sets)
    print(letter_set)
    for contour in contours_list[:30]:
        print(contour)
    for i, contour in enumerate(letter_set):
        cropped = letter_detector.put_letter_in_square(contour[0], img)
        cv2.imwrite(f"{output_directory}/rack/{i}_{contour[1]}_{contour[2]}_letter.png", cropped)

    # clf = letter_detector.get_classifier_from_file(clf_file_name)
    # predicted, prob_predicted, letters_file_names = letter_detector.predict_letters(clf, f"{output_directory}/rack")
    # for i, letter_file_name in enumerate(letters_file_names):
    #     prob_with_classes = list(zip(clf.classes_, prob_predicted[i]))
    #     prob_with_classes.sort(key=lambda tup: tup[1], reverse=True)
    #     print(f"{letter_file_name}, prob - {prob_with_classes}, pred - {predicted[i]}")

    # cv2.drawContours(img, contours_list, -1, (0, 255, 0), 3)
    # cv2.imwrite("img_with_contours.png", img)
    # cv2.imshow('Contours', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

