import cv2
import torch
from ultralytics import YOLO
from object_detector import ObjectDetector
from object_detector import DepthEstimator


# Function to draw bounding box and depth information
def draw_info(image, obj):
    bbox = obj["bbox"]
    depth = obj["depth"]
    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
    cv2.putText(image, f"{obj['class']}: {depth:.2f}m", (bbox[0], bbox[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


def main():
    # Load models
    detector = ObjectDetector("models/yolov8.pt")
    estimator = DepthEstimator("models/MiDaS_small.pt")

    # Initialize video capture
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame from camera
        ret, frame = cap.read()

        if not ret:
            print("Error capturing frame!")
            break

        # Detect objects
        detections = detector.detect_objects(frame.copy())

        # Process and draw information for each object
        for obj in detections:
            bbox = obj["bbox"]
            depth = estimator.estimate_depth(frame.copy(), bbox)
            obj["depth"] = depth
            draw_info(frame, obj)

        # Display the processed frame
        cv2.imshow("3D Object Detection", frame)

        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()