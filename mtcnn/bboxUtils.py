import numpy as np 
from PIL import Image
from mtcnn.utils import preprocess

def nms(boxes,overlap_threshold=0.5,mode="union"):
    """
    非最大值抑制　Non-maximum supperssion

    Arguments:
        boxes: 表示box的numpy array，一行代表一个box(x1,y1,x2,y2,score)
        overlap_threshold:重叠合并的阈值
        mode:"union" or "min"

    Returns:
        合并后的box
    """
    if len(boxes) == 0:
        return []

    pick = []
    x1,y1,x2,y2,score = [boxes[:,i] for i in range(5)]

    area = (x2 - x1 + 1.0) * (y2 - y1 + 1.0) # 每个边框的面积

    # increasing order
    ids = np.argsort(score) # 置信度升序排序

    while len(ids) > 0:

        last = len(ids) - 1
        i = ids[last] #置信度最高的边框的index

        pick.append(i) #先保留置信度最高的边框

        # 当前置信度最高的边框的为 => (x1[i],y1[i],x2[i],y2[i])
        # 找出余下边框的和最高置信度边框有重叠的区域
        ix1 = np.maximum(x1[i],x1[ids[:last]])
        iy1 = np.maximum(y1[i],y1[ids[:last]])

        ix2 = np.minimum(x2[i],x2[ids[:last]])
        iy2 = np.minimum(y2[i],y2[ids[:last]])

        # 重叠区域的宽和高，如果box和最高置信度边框的box没有重叠部分，这ｗ和h必有一个为０，
        # 这样不重叠的box的重叠面积计算就为０
        w = np.maximum(0.0,ix2 - ix1 + 1.0)
        h = np.maximum(0.0,iy2 - iy1 + 1.0)

        inter = w * h  # 重叠区域的面积

        # 重叠区域所占比例的计算
        if mode == "min": # min下，inter / mini()
            overlap = inter / np.minimum(area[i],area[ids[:last]])
        elif mode == "union": #　union, inter / (box1 + box2 - inter)
            overlap = inter / (area[i] + area[ids[:last]] - inter)

        # delete all boxes where overlap larger than the threshold
        ids = np.delete(ids,np.concatenate([[last],np.where(overlap > overlap_threshold)[0]]))

    return pick

def calibrate_box(bboxes,offsets):
    """
    使用网络预测得到的偏移量offsets校正box，使其能够更好的框出来人脸

    Arguments:
        bboxes: numpy array [n,5]
        offsets:numpy array [n,4]

    Returns:
        校正后的box,[n,5]
    """
    x1,y1,x2,y2 = [bboxes[:,i] for i in range(4)]
    w = x2 - x1 + 1.0
    h = y2 - y1 + 1.0
    w = np.expand_dims(w,1)
    h = np.expand_dims(h,1)

    # tx1,ty1,tx2,ty2 = [offsets[:,i] for i in range(4)]
    # x1_true = x1 + tx1 * w
    # y1_true = y1 + ty1 * h
    # x2_true = x2 + tx2 * w
    # y2_true = y2 + ty2 * h
    # 偏移量单位的为box的宽和高

    translation = np.hstack([w,h,w,h]) * offsets
    bboxes[:,0:4] = bboxes[:,0:4] + translation
    return bboxes

def convert_to_square(bboxes):
    """
    　将box校正为正方形，方便作为下一阶段的输入
    """
    square_bboxes = np.zeros_like(bboxes)
    x1,y1,x2,y2 = [bboxes[:,i] for i in range(4)]
    h = y2 - y1 + 1.0
    w = x2 - x1 + 1.0
    max_side = np.maximum(h,w)

    square_bboxes[:,0] = x1 + w * 0.5 - max_side * 0.5
    square_bboxes[:,1] = y1 + h * 0.5 - max_side * 0.5
    square_bboxes[:,2] = square_bboxes[:,0] + max_side - 1.0
    square_bboxes[:,3] = square_bboxes[:,1] + max_side - 1.0

    return square_bboxes

def correct_bboxes(bboxes, width, height):
    """Crop boxes that are too big and get coordinates
    with respect to cutouts.

    Arguments:
        bboxes: a float numpy array of shape [n, 5],
            where each row is (xmin, ymin, xmax, ymax, score).
        width: a float number.
        height: a float number.

    Returns:
        dy, dx, edy, edx: a int numpy arrays of shape [n],
            coordinates of the boxes with respect to the cutouts.
        y, x, ey, ex: a int numpy arrays of shape [n],
            corrected ymin, xmin, ymax, xmax.
        h, w: a int numpy arrays of shape [n],
            just heights and widths of boxes.

        in the following order:
            [dy, edy, dx, edx, y, ey, x, ex, w, h].
    """

    x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
    w, h = x2 - x1 + 1.0,  y2 - y1 + 1.0

    num_boxes = bboxes.shape[0]

    # 'e' stands for end
    # (x, y) -> (ex, ey)
    x, y, ex, ey = x1, y1, x2, y2

    # we need to cut out a box from the image.
    # (x, y, ex, ey) are corrected coordinates of the box
    # in the image.
    # (dx, dy, edx, edy) are coordinates of the box in the cutout
    # from the image.
    dx, dy = np.zeros((num_boxes,)), np.zeros((num_boxes,))
    edx, edy = w.copy() - 1.0, h.copy() - 1.0

    # if box's bottom right corner is too far right
    ind = np.where(ex > width - 1.0)[0]
    edx[ind] = w[ind] + width - 2.0 - ex[ind]
    ex[ind] = width - 1.0

    # if box's bottom right corner is too low
    ind = np.where(ey > height - 1.0)[0]
    edy[ind] = h[ind] + height - 2.0 - ey[ind]
    ey[ind] = height - 1.0

    # if box's top left corner is too far left
    ind = np.where(x < 0.0)[0]
    dx[ind] = 0.0 - x[ind]
    x[ind] = 0.0

    # if box's top left corner is too high
    ind = np.where(y < 0.0)[0]
    dy[ind] = 0.0 - y[ind]
    y[ind] = 0.0

    return_list = [dy, edy, dx, edx, y, ey, x, ex, w, h]
    return_list = [i.astype('int32') for i in return_list]

    return return_list

def get_image_crop(bounding_boxes,img,size=24):
    """
    得到box对应于原图像的的区域，作为下一个网络的输入

    """

    num_boxes = len(bounding_boxes)
    width,height = img.size

    [dy,edy,dx,edx,y,ey,x,ex,w,h] = correct_bboxes(bounding_boxes,width,height)
    img_boxes = np.zeros((num_boxes,3,size,size),"float32")

    for i in range(num_boxes):
        img_box = np.zeros((h[i],w[i],3),"uint8")

        img_array = np.asarray(img,"uint8")
        
        img_box[dy[i]:(edy[i] + 1),dx[i]:(edx[i] + 1),:] = img_array[y[i]:(ey[i] + 1),x[i]:(ex[i] + 1) :]

        # resize
        img_box = Image.fromarray(img_box)
        img_box = img_box.resize((size,size),Image.BILINEAR)
        img_box = np.asarray(img_box,"float32")

        img_boxes[i,:,:,:] =  preprocess(img_box)

    return img_boxes