
import torch
from torch.autograd import Variable
import math 
from PIL import Image
import numpy as np 
from mtcnn.utils import preprocess
from mtcnn.bboxUtils import nms,calibrate_box,convert_to_square,get_image_crop

def run_pnet(image,pnet,scale,threshold,gpu=True):
    """
    执行P-Net，生成候选框，并为每个候选框生成一个置信度（包含人脸的可能性），最后使用NMS合并重叠的候选框

    Arguments:
        image:输入的待检测图像
        pnet: P-Net
        scale: 图像宽和高的缩放比例因子
        threshold:是否包含人脸概率的阈值，小于该值则删除该候选框

    Returns:
        numpay array [n_boxes,9]
        ９－> 4个值是bouding_box的在原始图像上的位置，１个值表示边框包含人脸的置信度，４个值是bouding_box位置相对于真是人脸位置的偏移量
    """
    width,height = image.size # 原输入图像的宽和高
    
    sw,sh = math.ceil(width * scale),math.ceil(height * scale) # 当前输入图像的宽和高
    img = image.resize((sw,sh),Image.BILINEAR)
    img = np.asarray(img,"float32")

    img = preprocess(img)
    img = Variable(torch.FloatTensor(img),volatile=True)

    if gpu and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        gpu = False
        device = torch.device("cpu")

    img = img.to(device)
    pnet = pnet.to(device)

    offsets,probs = pnet(img)

    if gpu:
        probs = probs.detach().cpu().data.numpy()[0,1,:,:]
        offsets = offsets.detach().cpu().data.numpy()
    else:
        probs = probs.data.numpy()[0,1,:,:]
        offsets = offsets.data.numpy()


    boxes = genrate_anchor_bboxes(probs,offsets,scale,threshold)

    return boxes

def run_rnet(image,bounding_boxes,rnet,threshold,gpu=True):
    bounding_boxes = convert_to_square(bounding_boxes)
    bounding_boxes[:,0:4] = np.round(bounding_boxes[:,0:4])

    img_crops = get_image_crop(bounding_boxes,image,size=24)
    img_crops = Variable(torch.FloatTensor(img_crops),volatile=True)

    if gpu and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        gpu = False
        device = torch.device("cpu")

    img_crops = img_crops.to(device)
    rnet = rnet.to(device)

    offsets,probs = rnet(img_crops)

    if gpu:
        probs = probs.detach().cpu().data.numpy()
        offsets = offsets.detach().cpu().data.numpy()
    else:
        probs = probs.data.numpy()
        offsets = offsets.data.numpy()

    keep = np.where(probs[:,1] > threshold)[0] # 删除不符合的bbox
    bounding_boxes = bounding_boxes[keep]

    bounding_boxes[:,4] = probs[keep,1].reshape((-1,)) # 更新置信度为O-Net的预测结果
    offsets = offsets[keep]

    keep = nms(bounding_boxes,0.7)
    bounding_boxes = bounding_boxes[keep]

    bounding_boxes = calibrate_box(bounding_boxes[:,0:4],offsets[kee])
    bounding_boxes = convert_to_square(bounding_boxes)
    bounding_boxes[:,0:4] = np.round(bounding_boxes[:,0:4])

    return bounding_boxes

def run_onet(image,bounding_boxes,onet,threshold,gpu=True):
    img_crops = get_image_crop(bounding_boxes,image,size=48)
    if len(img_crops) == 0:
        return [],[]

    img_crops = Variable(torch.FloatTensor(img_crops),volatile=True)

    if gpu and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        gpu = False
        device = torch.device("cpu")

    img_crops = img_crops.to(device)
    onet = onet.to(device)

    landmarks,offsets,probs = onet(img_crops)

    if gpu:
        probs = probs.detach().cpu().data.numpy()[0,1,:,:]
        offsets = offsets.detach().cpu().data.numpy()
        landmarks = landmarks.detach().cpu().data.numpy()
    else:
        probs = probs.data.numpy()[0,1,:,:]
        offsets = offsets.data.numpy()
        landmarks = landmarks.data.numpy()

    keep = np.where(probs[:,1] > threshold)[0]
    bounding_boxes = bounding_boxes[keep]
    bounding_boxes[:,4] = probs[keep,1].reshape((-1,))
    offsets = offsets[keep]
    landmarks = landmarks[keep]

    # 计算landmark points
    width = bounding_boxes[:,2] - bounding_boxes[:,0] + 1.0
    height = bounding_boxes[:,3] - bounding_boxes[:,1] + 1.0
    xmin,ymin = bounding_boxes[:,0],bounding_boxes[:,1]
    landmarks[:,0:5] = np.expand_dims(xmin,1) + np.expand_dims(width,1) * landmarks[:,0:5]
    landmarks[:,5:10] = np.expand_dims(ymin,1) + np.expand_dims(height,1) * landmarks[:,5:10]

    bounding_boxes = calibrate_box(bounding_boxes[:,0:4],offsets)
    keep = nms(bounding_boxes,0.7,mode="min")
    bounding_boxes = bounding_boxes[keep]
    landmarks = landmarks[keep]

    return bounding_boxes,landmarks


def genrate_anchor_bboxes(probs,offsets,scale,threshold):
    """
    根据最后Feature Map生成对应的在原输入图像上的bounding_box

    Arguments:
        probs: anchor对应的区域包含人脸的概率
        offsets: 预测的人脸边框相对于人脸真是边框的偏移量
        scale:　当前图像对应于原图像的缩放比例
        threshold: 包含人脸可能性的阈值，过滤小于该阈值的anchor

    Returns:
        numpy array [n_boxes,9]
        ９－> 4个值是bouding_box的在原始图像上的位置，１个值表示边框包含人脸的置信度，４个值是bouding_box位置相对于真是人脸位置的偏移量
    """

    # P-Net的对图像的处理过程，相当于一个12x12的窗口在原图上以步长为２滑动
    stride = 2
    cell_size = 12

    inds = np.where(probs > threshold) # 过滤掉小于阈值
    
    if len(inds) == 0:
        return np.array([])
    
    # 取出大于阈值的偏移量
    tx1,ty1,tx2,ty2 = [offsets[0,i,inds[0],inds[1]] for i in range(4)]
    offsets = np.array([tx1,ty1,tx2,ty2])
    score = probs[inds[0],inds[1]]

    # 生成anchor 对应于原图的bounding_boxes
    bounding_boxes = np.vstack([
        np.round((stride * inds[1] + 1.0) / scale),
        np.round((stride * inds[0] + 1.0) / scale),
        np.round((stride * inds[1] + 1.0 + cell_size) / scale),
        np.round((stride * inds[0] + 1.0 + cell_size) / scale),
        score,offsets
    ]) 

    return bounding_boxes.T