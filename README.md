# General info
RPC API created using FastAPI for detecting letters based on image with scrabble board in it.
In|  Out
:-------------------------:|:-------------------------:
<img src="https://github.com/radosz99/scrabble-board-detector/blob/main/images/test.jpg" width=50% alt="Img"/>   |  <img src="https://github.com/radosz99/scrabble-board-detector/blob/main/images/output.png" width=100% alt="Img"/>



# Endpoints
<a name="best"></a>
## Get letters pattern (POST)
The only endpoint in API requires an image in request body.
### URL
```
http://127.0.0.1:8000/detect_letters
```
### Example using Python requests
In request there must be parameter `files` included, with key `image` and binary representation of image as value.
```
import requests
import ast

IMG_PATH = "test.jpg"
SERVER_URL = "http://127.0.0.1:8000"

files = {'image': open(IMG_PATH,'rb')}
response = requests.post(url=f"{SERVER_URL}/detect_letters", files=files)
json_response = ast.literal_eval(response.text)
board = json_response['board']
```
As a response there is a json with only one key-value pair - `board` and its representation as a list of list:
```
{
  "board": [[' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
	    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
	    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
	    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
	    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
	    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
	    [' ', 'o', 'p', 'e', 'r', 'a', 't', 'i', 'o', 'n', 'a', 'l', ' ', ' ', ' '],
	    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
	    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
	    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
	    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
	    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
	    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
	    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
	    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ']]
}
```

# Run
## FastAPI (Windows)
From the root folder:
```
$ uvicorn project.main:app --reload
```
# Principle
## Image with board in it
To detect something firstly is required to have an image with board (all edges required) in it:
<p align="center">
  <img src="https://github.com/radosz99/scrabble-board-detector/blob/main/images/test.jpg" width=40% alt="Img"/> 
</p> 

## Board detection in image
Then function `align_images` from `project/board_detector.py` firstly cutting out board from the image using some OpenCV stuff(`BRISK_create`, `detectAndCompute`, `DescriptorMatcher_create`, `findHomography`, `warpPerspective`) to this form:
<p align="center">
  <img src="https://github.com/radosz99/scrabble-board-detector/blob/main/images/board.png" width=40% alt="Img"/> 
</p> 

Secondly board is being thresholded using OpenCV `threshold` method to facilitate later analysis and also right board is extracted from board using manually calculated proportions:
```
BOARD_LEFT_UP_CORNER_HEIGHT = 0.048575
BOARD_RIGHT_DOWN_CORNER_HEIGHT = 0.949556
BOARD_LEFT_UP_CORNER_WIDTH = 0.081336
BOARD_RIGHT_DOWN_CORNER_WIDTH = 0.916524
```

<p align="center">
  <img src="https://github.com/radosz99/scrabble-board-detector/blob/main/images/board_after_threshold.png" width=40% alt="Img"/> 
</p> 

## Getting cells from the board
Then using function `divide_board_in_cells` from `project/letter_detector.py` board is divided in cells (15x15) which are saved to the files if they are 'white enough' which means if `check_if_cell_is_used` function returns `True`. That function checks average of pixels value in some random row and if it is too black it return False and cell is considered as not used. 'White enough' cells are saved to chosen directory like this with their coordinates in filename:
<p align="center">
  <img src="https://github.com/radosz99/scrabble-board-detector/blob/main/images/cells.png" width=85% alt="Img"/> 
</p> 

## Extracting letters from cells
Next stage relies on finding contours in cells that can be considered as letters. With function `leave_only_cells_with_contours_similar_to_letter` from `project/letter_detector.py` first all contours are extracted using OpenCV `findContours` method, then amount of this contours is checked (`check_if_valid_amount_of_contours`) and valid contour is resized and saved to file which name is the same as name of the file with cell from which contour is:
<p align="center">
  <img src="https://github.com/radosz99/scrabble-board-detector/blob/main/images/cleared_cells.png" width=85% alt="Img"/> 
</p> 

## Few words about training classifier
For classifier training was created script `train.py`. Training is nearly fully automated so the only thing that user must do is to take photos for every letter putted on board in many configurations. Photos should be placed in appropriate directories for every letter. Suppose that we created directory `training` in root directory for training workspace and inside it directory for every letter in alphabet. So project structure now should look like this:
```
scrabble-board-detector/
|
|── training/
|   |── a/
|   |── b/
|   |── c/
|   |── ...
|   |── y/
|   |── z/
|
|── classifiers/
|── images/
|── project/
|── tests/
|── resources/
|
```
Under each of 'letter directory' there should be images with board in it and board should contain only tiles with this letter, no matter how many, there can be all possible tiles with that letter. Filenames of images with board might in any convention (or no convention):
```
scrabble-board-detector/
|
|── training/
|   |── a/
|      |── board1.png
|      |── ...
|      |── board2.png
|   |── b/
|      |── 814814.jpg
|      |── ...
|      |── 82899949.png
|   |── ...
|   |── y/
|      |── brd.png
|      |── ...
|      |── brd0.png
|   |── z/
|      |── brd_with_z_1.png
|      |── ...
|      |── brd_with_z_67.png
|
|── classifiers/
|── images/
|── project/
|── tests/
|── resources/
|
```
It is basically all, last thing is run script `project/train.py` by:
```
$ python project/test.py
```
Calling `collect_samples_from_boards(TRAINING_WORKSPACE_DIR)` (`TRAINING_WORKSPACE_DIR` by default is `training`) from that script will prepare data for training:
 1. Goes through every directory in `TRAINING_WORKSPACE_DIR` and cut board from every image in letter directories and placed cutted boards in `TRAINING_WORKSPACE_DIR/{letter}/cropped_boards`, for example cutted board from `TRAINING_WORKSPACE_DIR/{letter}/{file_name}` will be saved as `TRAINING_WORKSPACE_DIR/{letter}/cropped_boards/{file_name}_cropped.png`. Function `training_utils.get_boards_from_images`
 2. In each directory in `TRAINING_WORKSPACE_DIR/{letter}/cropped_boards` directories where cells will be placed are created. For example `TRAINING_WORKSPACE_DIR/{letter}/cropped_boards/{file_name}_cropped_cells`
 3. All valid cells from each image are being saved to appropriate directory. Filename clearly indicates coordinates of the cell. Function `training_utils.divide_boards_in_cells`
 4. In each directory in format `TRAINING_WORKSPACE_DIR/{letter}/cropped_boards/{file_name}_cropped_cells` directories where valid contours that can be considered as a letter will be placed are created. For example `TRAINING_WORKSPACE_DIR/{letter}/cropped_boards/{file_name}_cropped_cells/cleared`.
 5. Then calling function `training_utils.leave_only_cells_probably_with_letter` will collect new cells in created directories.
 6. Finally function `training_utils.collect_letters_to_one_directory` will look for `cleared` directories in each letter directory and collect all letter samples to one directory:
 <p align="center">
  <img src="https://github.com/radosz99/scrabble-board-detector/blob/main/images/samples.png" width=80% alt="Img"/> 
</p> 

Calling `training_utils.get_trained_classifier(TRAINING_WORKSPACE_DIR)` will make a classifier based on prepared data:
 1. First it will collect data from all `samples` directories and reformat it appropriate to **Convolutional Neural Network**. All samples are now stored as 12x12 pixels image.
 2. Each sample is being converted to 4-bit grayscale (function `convert_image_to_4_bit_array`) and stored in vector 1x144 and each sample is putted to samples vector, so finally vector Ax1x144 is created where A is number of samples. Also vector with target (letters corresponding to vector at the same index) is being created with size Ax1.
 3. Then classifier is being initiated, train data is being generated and classifier is being tested (`TEST_SIZE` by default is 0.5):
 ```
 clf = svm.SVC(gamma=0.001, probability=True)
 X_train, _, y_train, _ = train_test_split(data, target, test_size=TEST_SIZE, shuffle=True)
 clf.fit(X_train, y_train)
 ```
 4. Next it can be saved in file using `training_utils.save_classifier_to_file`.
 
 
# How to recognize letters in image?
Calling `recognize_letters_from_image` from `project/train.py`
