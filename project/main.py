from project.training import recognize_letters_from_image
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
    board = recognize_letters_from_image(img_path=file_name, clf_file_name=CLASSIFIER_PATH)
    new_board = [row.tolist() for row in board]
    return {"board": new_board}


# training.collect_samples_from_boards('training')
# clf = training.get_trained_classifier('training')
# training.save_classifier_to_file(clf)
# training.resize_image(divider=4, img_file_name=IMG_FILE_NAME)
