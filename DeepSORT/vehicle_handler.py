import cv2
import numpy as np
from PIL import Image
from yolo import YOLO
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
import pandas as pd

class VehicleTracker():

    def __init__(self, model_file_name, columns_names=['Vehicle_No', 'Top_Left_x', 'Top_Left_y', 'Bottom_Right_x', 'Bottom_Right_y']):
        max_cosine_distance = 0.3
        nn_budget = None
        self.metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.nms_max_overlap = 1.0
        self.encoder = gdet.create_box_encoder(model_file_name, batch_size=1)
        self.columns_names = columns_names
        self.is_creating_overlay_frames = True
        self.yolo = YOLO()

    def initialize_new_tracker(self):
        self.tracker = Tracker(self.metric)

    def get_vehicles(self, image_path):
        detected_vehicles = {name: [] for name in self.columns_names}
        frame = cv2.imread(image_path)
        boxs = self.yolo.detect_image(Image.fromarray(frame))
        features = self.encoder(frame, boxs)
        detections = [Detection(box, 1.0, feature) for box, feature in zip(boxs, features)]
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        self.tracker.predict()
        self.tracker.update(detections)
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update <= 4:
                bbox = track.to_tlbr()
                detected_vehicles[self.columns_names[0]].append(track.track_id)
                detected_vehicles[self.columns_names[1]].append(int(bbox[0]))
                detected_vehicles[self.columns_names[2]].append(int(bbox[1]))
                detected_vehicles[self.columns_names[3]].append(int(bbox[2]))
                detected_vehicles[self.columns_names[4]].append(int(bbox[3]))
        return pd.DataFrame(detected_vehicles)

    def plot_detected_vehicles(self, frame, detected_vehicles):
        for index in detected_vehicles.index:
            cv2.rectangle(frame,
                          (detected_vehicles[self.columns_names[1]][index], detected_vehicles[self.columns_names[2]][index]),
                          (detected_vehicles[self.columns_names[3]][index], detected_vehicles[self.columns_names[4]][index]),
                          (0, 0, 255),
                          2)
            cv2.putText(frame,
                        str(detected_vehicles[self.columns_names[0]][index]),
                        (detected_vehicles[self.columns_names[1]][index], detected_vehicles[self.columns_names[2]][index]),
                        0,
                        5e-3 * 200,
                        (0, 0, 255),
                        2)
        return frame
