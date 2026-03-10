import cv2
import numpy as np
import kagglehub
import time
import os

from cv2 import dnn
from math import ceil

# FER 2013 training dataset through Kaggle, images are all in greyscale to denoise
# https://www.kaggle.com/datasets/msambare/fer2013
filename = kagglehub.dataset_download("msambare/fer2013")

# following geeksforgeeks tutorial
# https://learnopencv.com/facial-emotion-recognition/
# https://docs.opencv.org/4.x/db/d27/tutorial_py_table_of_contents_feature2d.html

# image definition variables
image_mean = np.array([127, 127, 127]) # centers the data around zero
image_std = 128.0 # squishes pixel values into range of approx. -1.0 to 1.0

# iou = intersection over union
iou_threshold = 0.3 # removes duplicate boxes that are overlapping the same object

# scaling factors to help model encode and decode box coordinates more stably
center_variance = 0.1
size_variance = 0.2

# tiny boxes (10, 16) find small objects and large boxes (256) find big objects
min_boxes = [10.0, 16.0, 24.0, 32.0, 48.0, 64.0, 96.0, 128.0, 192.0, 256.0]

# a stride of 8 means one pixel in the feature map represents an 8x8 block of pixels in the original image
strides = [8.0, 16.0, 32.0, 64.0] # represents how much the image is shrunk when passed through neural network
threshold = 0.5 # confidence score

def define_img_size(image_size):
    shrinkage_list = []
    feature_map_w_h_list = []
    for size in image_size:
        feature_map = [int(ceil(size / stride)) for stride in strides]
        feature_map_w_h_list.append(feature_map) # calculates how large internal grids will be after being divided by the strides
    for i in range(0, len(image_size)):
        shrinkage_list.append(strides)
    priors = generate_priors(
        feature_map_w_h_list, shrinkage_list, image_size, min_boxes
    ) # the search area templates the model uses to scan the image
    return priors

def generate_priors(feature_map_list, shrinkage_list, image_size, min_boxes):
    # function works in normalised co-ordinates (values between 0 and 1)
    priors = []

    # triple loop ensures every pixel covered
    # first loop iterates through different levels of detail
    # first looks for tiny objects using a small grid and last uses a coarse grid to look for big objects
    for index in range(0, len(feature_map_list[0])):
        scale_w = image_size[0] / shrinkage_list[0][index]
        scale_h = image_size[1] / shrinkage_list[1][index]
        
        # i and j move through each row and column across current grid
        for j in range(0, feature_map_list[0][index]):
            for i in range(0, feature_map_list[0][index]):

                # these are where the box is calculated
                x_center = (i + 0.5) / scale_w
                y_center = (j + 0.5) / scale_h

                # places multiple boxes of different sizes at every grid point
                for min_box in min_boxes[index]:

                    # these are high wide and tall the box is
                    w = min_box / image_size[0]
                    h = min_box / image_size[1]
                    priors.append([x_center, y_center, w, h])

    # np.clip() ensures no box exists outside of image boundaries
    print("priors nums:{}".format(len(priors)))
    return np.clip(priors, 0.0, 1.0)

# nms = non-maximum supression, deletes duplicate boxes and keeps best one
# hard because it deleted the boxes instantly, soft_nms just reduces score of overlapping boxes
def hard_nms(box_scores, iou_threshold, top_k = -1, candidate_size = 200):
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []

    # sort boxes by confidence
    indexes = np.argsort(scores)
    indexes = indexes[-candidate_size:]
    while len(indexes) > 0:
        current = indexes[-1]

        # best box gets added to picked list
        picked.append(current)
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[:-1]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(rest_boxes, np.expand_dims(current_box, axis = 0))

        # only keeps indexes where the overlap is low (below the threshold)
        indexes = indexes[iou <= iou_threshold]
    return box_scores[picked, :]

# if the bottom right is actually the top left, it forces area to be 0 instead of a negative number
def area_of(left_top, right_bottom):
    hw = np.clip(right_bottom - left_top, 0.0, None)
    return hw[..., 0] * hw[..., 1]

# finds out how much two boxes overlap
# eps is a tiny number added to the denominator to prevent a division by zero error if area is empty
def iou_of(boxes0, boxes1, eps = 1e-5):
    overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])
    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)

def predict(width, height, confidences, boxes, prob_threshold, iou_threshold = 0.3, top_k = -1):
    boxes = boxes[0]
    confidences = confidences[0]
    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis = 1)
        box_probs = hard_nms(box_probs, iou_threshold = iou_threshold, top_k = top_k)
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    return(picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4])

def convert_locations_to_boxes(locations, priors, center_variance,
                               size_variance):
    if len(priors.shape) + 1 == len(locations.shape):
        priors = np.expand_dims(priors, 0)
    return np.concatenate([locations[..., :2] * center_variance * priors[..., 2:] + priors[..., :2], np.exp(locations[..., 2:] * size_variance) * priors[..., 2:]], axis=len(locations.shape) - 1)

def center_form_to_corner_form(locations):
    return np.concatenate([locations[..., :2] - locations[..., 2:] / 2, locations[..., :2] + locations[..., 2:] / 2], len(locations.shape) - 1)

def FER_live_cam():
    emotion_dict = {
        0: 'neutral', 
        1: 'happiness', 
        2: 'surprise', 
        3: 'sadness',
        4: 'anger', 
        5: 'disgust', 
        6: 'fear'
    }

    video = cv2.VideoCapture(0)

    frame_width = int(video.get(3))
    frame_height = int(video.get(4))
    size = (frame_width, frame_height)

    # read ONNX model
    model = 'onnx_model.onnx'
    model = cv2.dnn.readNetFromONNX('emotion-ferplus-8.onnx')
    
    # read the Caffe face detector.
    model_path = 'RFB-320/RFB-320.caffemodel'
    proto_path = 'RFB-320/RFB-320.prototxt'
    net = dnn.readNetFromCaffe(proto_path, model_path)
    input_size = [320, 240]
    width = input_size[0]
    height = input_size[1]
    priors = define_img_size(input_size)

    while True:
        ret, frame = video.read()
        if ret:
            img_ori = frame

            # testing
            print("frame size: ", frame.shape)
            rect = cv2.resize(img_ori, (width, height))
            rect = cv2.cvtColor(rect, cv2.COLOR_BGR2RGB)
            net.setInput(dnn.blobFromImage(
                rect, 1 / image_std, (width, height), 127)
            )
            start_time = time.time()
            boxes, scores = net.forward(["boxes", "scores"])
            boxes = np.expand_dims(np.reshape(boxes, (-1, 4)), axis=0)
            scores = np.expand_dims(np.reshape(scores, (-1, 2)), axis=0)
            boxes = convert_locations_to_boxes(
                boxes, priors, center_variance, size_variance
            )
            boxes = center_form_to_corner_form(boxes)
            boxes, labels, probs = predict(
                img_ori.shape[1], 
                img_ori.shape[0], 
                scores, 
                boxes, 
                threshold
            )
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            for (x1, y1, x2, y2) in boxes:
                w = x2 - x1
                h = y2 - y1
                cv2.rectangle(frame, (x1,y1), (x2, y2), (255,0,0), 2)
                resize_frame = cv2.resize(
                    gray[y1:y1 + h, x1:x1 + w], (64, 64)
                )
                resize_frame = resize_frame.reshape(1, 1, 64, 64)
                model.setInput(resize_frame)
                output = model.forward()
                end_time = time.time()
                fps = 1 / (end_time - start_time)
                print(f"FPS: {fps:.1f}")
                pred = emotion_dict[list(output[0]).index(max(output[0]))]
                cv2.rectangle(
                    img_ori, 
                    (x1, y1), 
                    (x2, y2), 
                    (215, 5, 247), 
                    2,
                    lineType=cv2.LINE_AA
                )
                cv2.putText(
                    frame, 
                    pred, 
                    (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, 
                    (215, 5, 247), 
                    2,
                    lineType=cv2.LINE_AA
                )

            result.write(frame)
        
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    video.release() # release the video capture object
    cv2.destroyAllWindows() # close all OpenCV windows

if __name__ == "__main__":
    FER_live_cam()