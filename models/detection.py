"""
Detection module - handles pet body and face detection using YOLOv8
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Tuple
from pathlib import Path


class PetDetector:
    """Detects pets (dogs/cats) and their faces in images using YOLO"""

    def __init__(
        self, body_model_path: str = "yolov8n.pt", face_model_path: str = None
    ):
        """
        Initialize detector with YOLO models

        Args:
            body_model_path: Path to YOLOv8 model for pet body detection
            face_model_path: Path to YOLOv8 model for face detection (optional)
        """
        print(f"Loading body detection model: {body_model_path}")
        self.body_model = YOLO(body_model_path)

        # Face detection model (optional for MVP, we'll add later)
        self.face_model = None
        if face_model_path:
            print(f"Loading face detection model: {face_model_path}")
            self.face_model = YOLO(face_model_path)

    def detect_bodies(self, image_path: str, conf_threshold: float = 0.5) -> List[Dict]:
        """
        Detect pet bodies in an image

        Args:
            image_path: Path to image file
            conf_threshold: Confidence threshold (0-1)

        Returns:
            List of detections with bounding boxes and confidence scores
        """
        # Run inference
        results = self.body_model(image_path, conf=conf_threshold)

        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Filter for dogs (class 16) and cats (class 15) in COCO dataset
                class_id = int(box.cls[0])
                if class_id in [15, 16]:  # cat=15, dog=16 in COCO
                    detection = {
                        "type": "body",
                        "class": "cat" if class_id == 15 else "dog",
                        "confidence": float(box.conf[0]),
                        "bbox": {
                            "x": int(box.xyxy[0][0]),
                            "y": int(box.xyxy[0][1]),
                            "width": int(box.xyxy[0][2] - box.xyxy[0][0]),
                            "height": int(box.xyxy[0][3] - box.xyxy[0][1]),
                        },
                    }
                    detections.append(detection)

        return detections

    def crop_detection(self, image_path: str, bbox: Dict) -> np.ndarray:
        """
        Crop image region based on bounding box

        Args:
            image_path: Path to original image
            bbox: Bounding box dict with x, y, width, height

        Returns:
            Cropped image as numpy array
        """
        image = cv2.imread(image_path)
        x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
        cropped = image[y : y + h, x : x + w]
        return cropped

    def detect_and_crop(
        self, image_path: str, conf_threshold: float = 0.5
    ) -> Tuple[List[Dict], List[np.ndarray]]:
        """
        Detect pets and return both detection info and cropped images

        Args:
            image_path: Path to image file
            conf_threshold: Confidence threshold

        Returns:
            Tuple of (detections list, cropped images list)
        """
        detections = self.detect_bodies(image_path, conf_threshold)
        cropped_images = []

        for detection in detections:
            cropped = self.crop_detection(image_path, detection["bbox"])
            cropped_images.append(cropped)

        return detections, cropped_images


# Test function
def test_detector():
    """Test the detector on a sample image"""
    detector = PetDetector()

    # Test image path
    test_image = "data/test_images/dog.jpg"

    if not Path(test_image).exists():
        print(f"Test image not found: {test_image}")
        return

    print(f"\nTesting detection on: {test_image}")
    detections, cropped = detector.detect_and_crop(test_image)

    print(f"\nFound {len(detections)} pets:")
    for i, det in enumerate(detections):
        print(f"  {i+1}. {det['class']} - confidence: {det['confidence']:.2f}")
        print(f"     bbox: {det['bbox']}")

    # Save cropped images
    for i, crop in enumerate(cropped):
        output_path = f"data/test_images/cropped_{i}.jpg"
        cv2.imwrite(output_path, crop)
        print(f"  Saved cropped image: {output_path}")


if __name__ == "__main__":
    test_detector()
