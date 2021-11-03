import os
# Off debug information.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import sys
import numpy as np
from tensorflow.keras.models import load_model

LETTERS = list("АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ")
SIZE = 68


def get_contours(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 75, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (90, 90))
    # Dilate image to unambiguously determine contours.
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # Sort contours list by X-axis.
    return sorted(contours, key=lambda a: cv2.boundingRect(a)[0])


def get_images(img):
    images = []
    for cnt in get_contours(img):
        x, y, w, h = cv2.boundingRect(cnt)
        cropped = img[y:y + h, x:x + w]
        images.append(cropped)
    return images


def model_predict(model, img):
    img_arr = np.expand_dims(img, axis=0)
    img_arr = img_arr.reshape((1, SIZE, SIZE, 1))
    result = list(model.predict([img_arr])[0])
    return LETTERS[result.index(max(result))]


def normalize(image):
    # Image preprocessing to increase recognition accuracy.
    image = cv2.resize(image, (SIZE, SIZE), cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, image = cv2.threshold(image, 70, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    image = cv2.erode(image, kernel, iterations=1) / 255.0
    return image


def main(path_to_img, path_to_model):
    img = cv2.imread(path_to_img)
    model = load_model(path_to_model)
    result = [model_predict(model, normalize(image)) for image in get_images(img)]
    file = open(f"{path_to_img}.txt", "w+")
    file.write(''.join(result))
    file.close()
    print(f'Result moved to {path_to_img}.txt')


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("Error. Enter path to image.")
    elif len(sys.argv) == 2:
        main(sys.argv[1], 'model.h5')
    elif len(sys.argv) > 2:
        main(sys.argv[1], sys.argv[2])