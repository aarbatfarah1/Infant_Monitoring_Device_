import cv2
from ultralytics import YOLO
import supervision as sv

# List of sharp objects to detect
sharp_objects = ["bottle", "wine glass", "cup", "fork", "knife", "scissors", "hair drier"]

def detect_sharp_objects(detections, model):
    detected_sharp_objects = set()
    for class_id in detections.class_id:
        object_name = model.model.names[class_id]
        if object_name in sharp_objects:
            detected_sharp_objects.add(object_name)
    return list(detected_sharp_objects)

def process_frame(frame, model):
    result = model(frame)[0]
    detections = sv.Detections.from_ultralytics(result)

    labels = [
        f"{model.model.names[class_id]} {confidence:0.2f}"
        for class_id, confidence in zip(detections.class_id, detections.confidence)
    ]

    detected_sharp_objects = detect_sharp_objects(detections, model)

    if detected_sharp_objects:
        print(f"Sharp objects detected: {', '.join(detected_sharp_objects)}")

    return labels, detections, detected_sharp_objects

def main(model, image_path):
    try:
        print("Loading image...")
        frame = cv2.imread(image_path)

        if frame is None:
            print(f"Error: Could not load image from {image_path}")
            return

        print("Processing image...")
        labels, detections, detected_sharp_objects = process_frame(frame, model)

        # Count detections for each class
        person_count = 0  # Initialize person count for the image

        for label in labels:
            class_name, confidence = label.split(' ')
            if class_name == "person":
                person_count += 1

        print(f"Persons: {person_count}")
        print(f"Detected sharp objects: {detected_sharp_objects}")

    except Exception as e:
        print(f"Error occurred during processing: {e}")
    return person_count, detected_sharp_objects

if __name__ == "__main__":
    image_path = "sharp.jpg"  # Replace with your image path
    model = YOLO('yolov9e.pt')
    nb_pers, sharp_objects = main(model, image_path)
    
