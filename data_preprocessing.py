import os
import onnx
import onnxruntime as ort
from onnx_tf.backend import prepare
import cv2
import numpy as np
svm_model_name ="/home/ubuntu/Desktop/GTAA/model_backup/fr_svm.h5"
encoder_file_path = "/home/ubuntu/Desktop/GTAA/model_backup/encoder.pkl"
im_file_path = "/home/ubuntu/Desktop/GTAA/model_backup/image_file_name.pkl"
path ="/home/ubuntu/Documents/image/"
pickle_path = "/home/ubuntu/Desktop/GTAA/model_backup/fac_rec_vector.pkl"
onnx_model_path = "/home/ubuntu/Desktop/GTAA/ultra_light/ultra_light_640.onnx"
def area_of(left_top, right_bottom):

    hw = np.clip(right_bottom - left_top, 0.0, None)
    return hw[..., 0] * hw[..., 1]
def iou_of(boxes0, boxes1, eps=1e-5):

    overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)
def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):

    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    indexes = np.argsort(scores)
    indexes = indexes[-candidate_size:]
    while len(indexes) > 0:
        current = indexes[-1]
        picked.append(current)
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[:-1]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(
            rest_boxes,
            np.expand_dims(current_box, axis=0),
        )
        indexes = indexes[iou <= iou_threshold]

    return box_scores[picked, :]
def predict(width, height, confidences, boxes, prob_threshold, iou_threshold=0.5, top_k=-1):

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
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = hard_nms(box_probs,
           iou_threshold=iou_threshold,
           top_k=top_k,
           )
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]

def fetch_image(path):
    folder = os.listdir(path)
    image_dict = {}
    image_label_list = []
    for j in folder:
        list_of_images = os.listdir(path+j)
        print(list_of_images)

        collection = []
        print(j)
        for i in range(len(list_of_images)):
            print(list_of_images[i])
            img_path = path+j+"/"+list_of_images[i]
            img = cv2.imread(path+j+"/"+list_of_images[i])
            label = j
            
            image_label_list.append([img,j,img_path])
    return image_label_list

def resize_image(image,width,height):
    h,w,_ = image.shape
    #img = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    img = cv2.resize(image,(width,height))
    img_mean = np.array([127, 127, 127])
    img = (img - img_mean) / 128
    img = np.transpose(img, [2, 0, 1])
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)
    return img
folder = os.listdir(path) 
image_dict = {}
print(folder)
img_list = []
onnx_model = onnx.load(onnx_model_path)

predictor = prepare(onnx_model)
ort_session = ort.InferenceSession(onnx_model_path)
input_name = ort_session.get_inputs()[0].name

for j in folder:
    list_of_images = os.listdir(path+j)
    print(path+j)
    croped_path = "/home/ubuntu/Documents/image/" + j+"_crop"
    os.mkdir(croped_path)
    for i in range(len(list_of_images)):
        print(list_of_images[i])
        
        picture  = cv2.imread(path+j+"/"+list_of_images[i])
           

        h, w,_ = picture.shape
        img  = resize_image(picture,640,480)
        confidences, boxes = ort_session.run(None, {input_name: img})
        boxes, labels, probs = predict(w, h, confidences, boxes, 0.7)
        for box in boxes:
            x1,y1,x2,y2 = box
            crop_image = picture[y1:y2,x1:x2]
            try:
                cv2.imwrite(croped_path+"/"+list_of_images[i],crop_image)
            except:
                print("Can't be croped")
            

    