
# ----------------------------------------------
# --- Author         : Anıl D. ŞENEL - 140502039
# --- Author         : Eren SAÇLI - 140502005
# --- Author         : Nazelin ÖZALP - 140502023
# --- Date           : 27th April 2019
# ----------------------------------------------

# Imports
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
import numpy as np
import csv
import time

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

# Object detection imports
from utils import label_map_util
from utils import visualization_utils as vis_util


if tf.__version__ < '1.4.0':
    raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!'
                      )

import datetime

CURRENT_DT = datetime.datetime.now()
savePath = 'detected_vehicles/'+str(CURRENT_DT.year)+'-'+str(CURRENT_DT.month)+'-'+str(CURRENT_DT.day)+'/'
#check image directory exit
if(os.path.exists(savePath)==False):
    os.mkdir(savePath)
print('WELCOME TO ANE -CAR COUNTING SYSTEM!')
video_path = input('ANTER THE VIDEO NAME OR PATH: ')
if(os.path.exists(video_path) == False):
    print('VIDEO COULD NOT BE FOUND!')
    exit()
# input video
cap = cv2.VideoCapture(video_path)

# Variables
total_passed_vehicle = 0  # using it to count vehicles

# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90
# Download Model
# uncomment if you have not download the model yet
# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# Loading label map
# Label maps map indices to category names, so that when our convolution network predicts 5, we know that this corresponds to airplane. Here I use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map,
        max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def start_app():
    total_passed_vehicle = 0
    ROI_RIGHT = int(input('ENTER THE ROI RIGHT COORNATE: '))
    ROI_TOP = int(input('ENTER THE ROI TOP COORNATE: '))
    ROI_BOTTOM = int(input('ENTER THE ROI BOTTOM COORNATE: '))
    ROI_LEFT = int(input('ENTER THE ROI LEFT COORNATE: '))
    ERROR_FACTOR_BT = int(input('ENTER THE ERROR FOCTOR VALUE FOR BOTTOM/TOP: '))
    ERROR_FACTOR_RL = int(input('ENTER THE ERROR FACTOR VALUE FOR RIGHT/LEFT: '))
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:

            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            csv_file = open('abc.csv','w')
            csv_file.close()
            imcodenumber=0 #number of the frame in the video.
            # for all the frames that are extracted from input video
            while cap.isOpened():
                (ret, frame) = cap.read() 
                #red means that if there is no  reading image image the param returns 0(false), if exist returns 1 (true)
                #frame params means that if there is an image the returned image assing it.
                imcodenumber+=1
                if not ret:
                    print('AND OF THE VIDEO, THANKS FOR USING TO THE APP!')
                    break
                
                input_frame = frame

                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(input_frame, axis=0)

                # Actual detection.
                (boxes, scores, classes, num) = \
                    sess.run([detection_boxes, detection_scores,
                             detection_classes, num_detections],
                             feed_dict={image_tensor: image_np_expanded})
                
                # Visualization of the results of a detection.
                #counter params means that if any car pass from the detexted ROI line the params returns number of the passing car.
                #csv_line params means that returns the basic info of the car in the CSV line.
                (counter) = \
                    vis_util.visualize_boxes_and_labels_on_image_array(
                    cap.get(1),
                    input_frame,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    roi_left=ROI_LEFT,
                    roi_right=ROI_RIGHT,
                    roi_top=ROI_TOP,
                    roi_bottom=ROI_BOTTOM,
                    error_factor_bt=ERROR_FACTOR_BT,
                    error_factor_rl=ERROR_FACTOR_RL,
                    use_normalized_coordinates=True,
                    line_thickness=4
                    )

                total_passed_vehicle = total_passed_vehicle + counter
              
                # insert information text to video frame
                # param 1: image, 
                # param 2: text, 
                # param 3: left-button cordinate,
                # param 4: font size, param 5: , 
                #param 6: color, param 7: text thickness, line type
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(
                    input_frame,
                    'Passing Vehicles: ' + str(total_passed_vehicle),
                    (10, 35),
                    font,
                    0.8,
                    (0, 255, 255),
                    2,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    )

                #when the vehicle passed over line and counted, make the color of ROI line green
                #param 1:image
                #param 2: right-top coordinate
                #param 4: left button coordinate.
                #param 5: color line,
                #param 6: border thickness
                if counter == 1:
                    cv2.rectangle(input_frame, (ROI_RIGHT, ROI_TOP), (ROI_BOTTOM, ROI_RIGHT), (255,255,255), 6)
                else:
                    cv2.rectangle(input_frame, (ROI_RIGHT, ROI_TOP), (ROI_LEFT,ROI_BOTTOM), (0, 0, 0), 6)

               
                cv2.imshow('ANE Vehicle Counting System', input_frame)

                #when 'q' later is typed exit from the app.
                #ord() funtion returns the ASCI code of typing button.
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break

            cap.release()
            cv2.destroyAllWindows()
    #os.remove('abc.csv')
start_app()