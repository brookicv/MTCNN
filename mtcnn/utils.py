import torch
import numpy as np 

from PIL import ImageDraw

def preprocess(img):
    img = img.transpose((2,0,1))
    img = np.expand_dims(img,0)
    img = (img - 127.5) * 0.0078125
    return img


def show_bboxes(img,bounding_boxes,facial_landmarks=[]):
    img_copy = img.copy()

    draw = ImageDraw.Draw(img_copy)

    for b in bounding_boxes:
        draw.rectangle([(b[0],b[1]),(b[2],b[3])],outline="red")

    for p in facial_landmarks:
        for i in range(5):
            draw.ellipse([(p[i] - 1.0,p[i + 5] - 1.0),(p[i] + 1.0,p[i+5] + 1.0)],outline="blue")

    return img_copy