# -*- coding: utf-8 -*-

import numpy as np 
import matplotlib.pyplot as plt 
from PIL import Image 
from mtcnn.model import PNet,RNet,ONet,load_weight
from mtcnn.run import run_pnet,run_rnet,run_onet
from mtcnn.utils import show_bboxes
from mtcnn.bboxUtils import nms,calibrate_box,convert_to_square,get_image_crop,correct_bboxes

import torch 
from torch.autograd import Variable


def build_pyramid(image,min_face_size=20.0):
    width,height = image.size 
    min_length = min(width,height)

    min_detection_size = 12
    factor = 0.707

    scales = []

    m = min_detection_size / min_face_size 
    min_length *= m 

    factor_count = 0
    while min_length > min_detection_size:
        scales.append(m * factor **factor_count)
        min_length *= factor 
        factor_count += 1

    return scales

pnet_weight = "weights/pnet.npy"
p_net = PNet()
load_weight(p_net,pnet_weight)

file = "1.jpg"
image = Image.open(file)

scales = build_pyramid(image)

bounding_boxes = []
for s in scales:
    boxes = run_pnet(image,p_net,s,0.6)
    bounding_boxes.append(boxes)

# 经过P-Net 后，所有可能包含人脸的box
bounding_boxes = [i for i in bounding_boxes if i is not None]
bounding_boxes = np.vstack(bounding_boxes)

print("经过P-Net 后，所有可能包含人脸的box个数:{}".format(bounding_boxes.shape[0]))

img1 = show_bboxes(image,bounding_boxes[:,0:4])

keep = nms(bounding_boxes,0.7)
bounding_boxes = bounding_boxes[keep]
print("NMS删除重叠较多的box后，box个数:{}".format(bounding_boxes.shape[0]))

img2 = show_bboxes(image,bounding_boxes[:,0:5])

bounding_boxes = calibrate_box(bounding_boxes,bounding_boxes[:,5:])

img3 = show_bboxes(image,bounding_boxes[:,0:4])


plt.figure()
plt.subplot(221)
plt.imshow(img1)

plt.subplot(222)
plt.imshow(img2)

plt.subplot(223)
plt.imshow(img3)


#####
###　STAGE 2
####
rnet = RNet()
load_weight(rnet,"weights/rnet.npy")
bounding_boxes = run_rnet(image,bounding_boxes,rnet,0.7)

onet = ONet()
load_weight(onet,"weights/onet.npy")
bounding_boxes,landmarks = run_onet(image,bounding_boxes,onet,0.8)

img4 = show_bboxes(image,bounding_boxes,landmarks)

plt.subplot(224)
plt.imshow(img4)
plt.show()
