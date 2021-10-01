import cv2
import numpy as np

GOOD_MATCH_PERCENT = 0.1

BOARD_LEFT_UP_CORNER_HEIGHT = 0.048575
BOARD_RIGHT_DOWN_CORNER_HEIGHT = 0.949556
BOARD_LEFT_UP_CORNER_WIDTH = 0.081336
BOARD_RIGHT_DOWN_CORNER_WIDTH = 0.916524


def align_images(im1, im2):
    # https://learnopencv.com/feature-based-image-alignment-using-opencv-c-python/
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    bri = cv2.BRISK_create()
    keypoints1, descriptors1 = bri.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = bri.detectAndCompute(im2Gray, None)

    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    matches.sort(key=lambda x: x.distance, reverse=False)

    matches = matches[:int(len(matches) * GOOD_MATCH_PERCENT)]
  
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    h, _ = cv2.findHomography(points1, points2, cv2.RANSAC)

    height, width, _ = im2.shape
    img = cv2.warpPerspective(im1, h, (width, height))
    cv2.imwrite(f"xdd.png", img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    return img

def retrieve_board_inside(board):
    height, width = board.shape
    start_height = int(BOARD_LEFT_UP_CORNER_HEIGHT * height)
    end_height = int(BOARD_RIGHT_DOWN_CORNER_HEIGHT * height)
    start_width = int(BOARD_LEFT_UP_CORNER_WIDTH * width)
    end_width = int(BOARD_RIGHT_DOWN_CORNER_WIDTH * width)
    return board[start_height:end_height, start_width:end_width]  # first height range, second width range

def get_board_from_image(ref_img_path, board_img_path):
    ref_img = cv2.imread(ref_img_path, cv2.IMREAD_COLOR)
    board_img = cv2.imread(board_img_path, cv2.IMREAD_COLOR)
  
    board = align_images(board_img, ref_img)
    return retrieve_board_inside(board)

