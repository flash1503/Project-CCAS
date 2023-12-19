import matplotlib.pyplot as plt
import glob
import os
import tqdm
import cv2
import configuration as config
from LaneNet.lane_handler import Lane_Detection
from DeepSORT.vehicle_handler import VehicleTracker
from Helper.ebs_handler import EmergencyBrakeEventDetector

if __name__ == "__main__":
    lane_detection = Lane_Detection(config.laneNet_weight_path)
    vehicle_tracker = VehicleTracker(config.deepSORT_weight_path)
    emergencyBrakeEventDetector = EmergencyBrakeEventDetector()

    sub_dirs = [o for o in os.listdir(config.input_image_dir) if os.path.isdir(os.path.join(config.input_image_dir, o))]
    skip_factor = 5

    right_to_ego_img = cv2.imread(os.path.join(config.sign_image_dir, 'turn-left.png'))
    left_to_ego_img = cv2.imread(os.path.join(config.sign_image_dir, 'turn-right.png'))
    stop_img = cv2.imread(os.path.join(config.sign_image_dir, 'stop.png'))

    for sub_dir in sub_dirs:
        vehicle_tracker.initialize_new_tracker()
        emergencyBrakeEventDetector.initialize_new_tracker()

        output_image_sub_dir = os.path.join(config.output_image_dir, sub_dir)
        output_video_sub_dir = os.path.join(config.output_video_dir, sub_dir)
        os.makedirs(output_image_sub_dir, exist_ok=True)
        os.makedirs(output_video_sub_dir, exist_ok=True)

        image_list = glob.glob(os.path.join(config.input_image_dir, sub_dir, '*.jpeg'), recursive=False)
        height, width, layers = cv2.imread(image_list[0]).shape
        video = cv2.VideoWriter(os.path.join(output_video_sub_dir, config.video_name), 0, 4, (width, height))

        for index, image_path in tqdm.tqdm(enumerate(image_list), total=len(image_list)):
            if index % skip_factor != (skip_factor - 1):
                continue
            print(f'Input Image Path : {image_path}')
            frame = cv2.imread(image_path)

            detected_vehicles = vehicle_tracker.get_vehicles(image_path)
            frame = vehicle_tracker.plot_detected_vehicles(frame, detected_vehicles)

            all_lane_dict = lane_detection.test_lanenet(image_path)
            y_list, all_x_list = lane_detection.extrapolate_lanes(all_lane_dict)
            ego_left_x_list, ego_right_x_list = lane_detection.ego_lane_extraction(y_list, all_x_list)
            ego_left_x_list, ego_right_x_list = lane_detection.ego_lane_tracking(ego_left_x_list, ego_right_x_list, ego_left_x_list, ego_right_x_list)
            frame = lane_detection.plot_ego_path(frame, y_list, ego_left_x_list, ego_right_x_list)

            cv2.putText(frame, os.path.basename(image_path), (500, 30), 0, 5e-3 * 200, (0, 0, 255), 5)

            emergency_brake_event_details = emergencyBrakeEventDetector.detect_emergency_brake_event(y_list, ego_left_x_list, ego_right_x_list, detected_vehicles)

            if any(emergency_brake_event_details[0].values()):
                for vehicle_no, direction in emergency_brake_event_details[0].items():
                    if int(detected_vehicles[detected_vehicles['Vehicle_No'] == vehicle_no]['Bottom_Right_y']) > 0:
                        if direction == 'Left':
                            frame[50:178, 200:328, :] = left_to_ego_img
                        elif direction == 'Right':
                            frame[50:178, 952:1080, :] = right_to_ego_img

            for emergency_brake_event_detail in emergency_brake_event_details[1]:
                print(f'Vehicle no {emergency_brake_event_detail[0]} came from {emergency_brake_event_detail[1]} side during frame no {index - (emergency_brake_event_detail[2] * skip_factor)} to frame no {index}')

            for vehicle_no in emergency_brake_event_details[2]:
                if int(detected_vehicles[detected_vehicles['Vehicle_No'] == vehicle_no]['Bottom_Right_y']) > 300:
                    frame[50:178, 576:704, :] = stop_img
                    print(f'Vehicle no {vehicle_no} is close to us')

            plt.figure('Detected Vehicles and Ego Lanes')
            plt.imshow(frame[:, :, (2, 1, 0)])
            plt.show(block=False)
            plt.pause(2)
            plt.close()
            cv2.imwrite(os.path.join(output_image_sub_dir, os.path.basename(image_path)), frame)
            print(f'Output Image Path : {os.path.join(output_image_sub_dir, os.path.basename(image_path))}')
            video.write(frame)

        cv2.destroyAllWindows()
        video.release()
        print(f'Output Video Path : {os.path.join(output_image_sub_dir, os.path.basename(image_path))}')
