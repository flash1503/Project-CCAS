from flask import Flask, request, jsonify
import cv2
from PIL import Image
from yolo import YOLO
from deep_sort import preprocessing, nn_matching
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

app = Flask(__name__)

def Counting_People(path):
    yolo = YOLO()
    unique_tracks = set()
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    video_capture = cv2.VideoCapture(path)
    w = int(video_capture.get(3))
    h = int(video_capture.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('output.avi', fourcc, 15, (w, h))
    list_file = open('detection.txt', 'w')
    frame_index = -1
    fps = 0.0

    while True:
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
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
                cv2.putText(frame, str(track.track_id), (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 200, (0, 255, 0), 2)
                unique_tracks.add(track.track_id)

        for det in detections:
            bbox = det.to_tlbr()
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)

        cv2.imshow('', frame)
        out.write(frame)
        frame_index += 1
        list_file.write(str(frame_index) + ' ')
        if boxs:
            for i in range(len(boxs)):
                list_file.write(' '.join(map(str, boxs[i])) + ' ')
        list_file.write('\n')
        fps = (fps + (1. / (time.time() - t1))) / 2
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    out.release()
    list_file.close()
    cv2.destroyAllWindows()
    return len(unique_tracks)

@app.route('/People_Count', methods=['POST'])
def postJsonHandler():
    content = request.get_json()
    path = content['path1']
    total = Counting_People(path)
    return jsonify({'Total number of people': total})

if __name__ == '__main__':
    app.run(debug=True)
