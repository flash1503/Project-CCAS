import sys
from yolo import YOLO, detect_video

if __name__ == '__main__':
    if len(sys.argv) < 2:
        exit()

    video_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    detect_video(YOLO(), video_path, output_path) if output_path else detect_video(YOLO(), video_path)
