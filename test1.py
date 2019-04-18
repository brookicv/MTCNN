# -*- coding: utf-8 -*-

from PIL import Image 
from mtcnn.model import PNet,RNet,ONet,load_weight
from mtcnn.runPNet import run_pnet




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

file = "/home/liqiang/445.jpeg"
image = Image.open(file)

scales = build_pyramid(image)

boxes = run_pnet(image,p_net,scales[2],0.7)

print(boxes)
