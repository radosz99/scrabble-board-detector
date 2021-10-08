from project.letter_detector import recognize_letters_from_image
from project.training_utils import resize_image
from fastapi import FastAPI, File, UploadFile
import os

CLASSIFIER_PATH = 'classifiers/final_clf.sav'
IMG_FILE_NAME = 'resources/test.jpg'

app = FastAPI()

@app.post("/detect_letters")
async def image(image: UploadFile = File(...)):
    if(not os.path.exists('tmp')):
        os.makedirs('tmp')
    file_name = os.getcwd() + "/tmp/" + image.filename.replace(" ", "-")
    with open(file_name,'wb+') as f:
        f.write(image.file.read())
        f.close()
    resize_image(divider=2, img_file_name=file_name, new_file_name=file_name)
    board = recognize_letters_from_image(img_path=file_name, clf_file_name=CLASSIFIER_PATH)
    new_board = [row.tolist() for row in board]
    return {"board": new_board}

