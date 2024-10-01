import cv2
from ultralytics import YOLO
import argparse
import json
import supervision as sv

sharp_objects = ["bottle", "wine glass", "cup", "fork", "knife", "scissors", "hair drier"]

def process_frame(frame, model, box_annotator):
    result = model(frame)[0]
    detections = sv.Detections.from_ultralytics(result)

    labels = [
        f"{model.model.names[class_id]} {confidence:0.2f}"
        for class_id, confidence in zip(detections.class_id, detections.confidence)
    ]

    sharp_object_detected = None
    for class_id in detections.class_id:
        object_name = model.model.names[class_id]
        if object_name in sharp_objects:
            sharp_object_detected = object_name
            break

    if sharp_object_detected:
        print(f"Sharp object detected: {sharp_object_detected}")

    annotated_frame = box_annotator.annotate(
        scene=frame, 
        detections=detections, 
        labels=labels
    )

    return annotated_frame, labels, detections

def main(model):
    try:
        args = parse_arguments()
        config = read_config(args.config)

        print("Opening the camera...")
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Could not open camera.")
            return

        print("Setting up annotator...")
        box_annotator = sv.BoxAnnotator(
            thickness=2,
            text_thickness=2,
            text_scale=1
        )

        while True:
            ret, frame = cap.read()

            if not ret:
                print("Error: Could not read frame.")
                break

            print("Processing frame...")
            annotated_frame, labels, detections = process_frame(frame, model, box_annotator)

            # Count detections for each class
            person_count = 0  # Initialize person count for each frame
            car_count = 0  # Initialize car count for each frame
            airplane_count = 0  # Initialize airplane count for each frame

            for label in labels:
                class_name, confidence = label.split(' ')
                if class_name == "person":
                    person_count += 1
                

            # Display counts on the frame
            count_text = f"Persons: {person_count}"
            cv2.putText(annotated_frame, count_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("Detection", annotated_frame)

            if cv2.waitKey(1) & 0xFF ==ord('q'):  # Wait for 'Esc' key to exit
                break

    except Exception as e:
        print(f"Error occurred during processing: {e}")

    finally:
        cap.release()
        cv2.destroyAllWindows()

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv9 live")
    parser.add_argument("--config", default="config.json", help="Path to configuration file")
    args = parser.parse_args()
    return args

def read_config(file_path: str):
    with open(file_path, 'r') as f:
        config = json.load(f)
    return config

if __name__ == "__main__":
    model = YOLO('yolov9e.pt')
    main(model)
