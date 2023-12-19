from __future__ import division, print_function, absolute_import

import os
from timeit import time
import warnings
import sys
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO
from cv2 import CAP_PROP_FRAME_COUNT
import imutils
import csv

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

warnings.filterwarnings('ignore')

def Counting_People(yolo, video_path=0):
    track_list = []
    total = 0
    currentFrame = 0
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0

    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    writeVideo_flag = True

    video_capture = cv2.VideoCapture(video_path)

    if writeVideo_flag:
        w = int(video_capture.get(3))
        h = int(video_capture.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter('output.avi', fourcc, 15, (w, h))
        list_file = open('detection.txt', 'w')
        frame_index = -1

    fps = 0.0
    for frame_number, _ in enumerate(iter(int, 1), 1):
        ret, frame = video_capture.read()
        if not ret:
            break
        t1 = time.time()
        image = Image.fromarray(frame)
        boxs = yolo.detect_image(image)
        features = encoder(frame, boxs)
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        tracker.predict()
        tracker.update(detections)

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update <= 1:
                bbox = track.to_tlbr()
                list1 = [frame_number, track.track_id] + list(map(int, bbox)) + [-1, -1, -1, -1]
                with open("res.txt", "a") as fp:
                    wr = csv.writer(fp, dialect='excel')
                    wr.writerow(list1)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
                cv2.putText(frame, str(track.track_id), (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 200, (0, 255, 0), 2)
                track_list.append(track.track_id)

        for det in detections:
            bbox = det.to_tlbr()
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)

        cv2.imshow('', frame)

        if writeVideo_flag:
            out.write(frame)
            frame_index += 1
            list_file.write(f"{frame_index} {' '.join(map(str, boxs.flatten()))}\n")

        fps = (fps + (1. / (time.time() - t1))) / 2
        total = len(set(track_list))
        print(f"fps= {fps:.2f}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    if writeVideo_flag:
        out.release()
        list_file.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    Counting_People(YOLO())
