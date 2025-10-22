"""
Image quality module - checks if images are suitable for pet detection
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple


class ImageQualityChecker:
    """Checks image quality for pet detection"""

    def __init__(
        self,
        min_brightness: float = 40.0,
        max_brightness: float = 220.0,
        min_sharpness: float = 30.0,
        min_resolution: int = 300,
    ):
        """
        Initialize quality checker

        Args:
            min_brightness: Minimum average brightness (0-255)
            max_brightness: Maximum average brightness (0-255)
            min_sharpness: Minimum Laplacian variance (sharpness)
            min_resolution: Minimum width/height in pixels
        """
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.min_sharpness = min_sharpness
        self.min_resolution = min_resolution

    def check_brightness(self, image: np.ndarray) -> Tuple[bool, float]:
        """
        Check if image is too dark or too bright

        Args:
            image: OpenCV image (BGR)

        Returns:
            (is_ok, brightness_value)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Calculate average brightness
        brightness = np.mean(gray)

        is_ok = self.min_brightness <= brightness <= self.max_brightness

        return is_ok, float(brightness)

    def check_sharpness(self, image: np.ndarray) -> Tuple[bool, float]:
        """
        Check if image is too blurry using Laplacian variance

        Args:
            image: OpenCV image (BGR)

        Returns:
            (is_ok, sharpness_value)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Calculate Laplacian variance (measure of sharpness)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()

        is_ok = sharpness >= self.min_sharpness

        return is_ok, float(sharpness)

    def check_resolution(self, image: np.ndarray) -> Tuple[bool, Tuple[int, int]]:
        """
        Check if image resolution is sufficient

        Args:
            image: OpenCV image

        Returns:
            (is_ok, (height, width))
        """
        height, width = image.shape[:2]

        is_ok = height >= self.min_resolution and width >= self.min_resolution

        return is_ok, (height, width)

    def detect_bars_or_cage(self, image: np.ndarray) -> bool:
        """
        Detect if image has cage bars or fence that might obscure pet

        Returns:
            True if bars/cage detected
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect edges
        edges = cv2.Canny(gray, 50, 150)

        # Use Hough Line Transform to detect strong vertical/horizontal lines
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10
        )

        if lines is None:
            return False

        # Count strong vertical lines (typical of kennel bars)
        vertical_lines = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

            # Vertical lines (close to 90 degrees)
            if 80 < angle < 100:
                vertical_lines += 1

        # If many vertical lines detected, likely a cage
        return vertical_lines > 5

    def check_quality(self, image_path: str) -> Dict[str, Any]:
        """Check if image quality is sufficient for matching"""
        img = cv2.imread(image_path)

        if img is None:
            return {"is_good": False, "warnings": ["Could not read image"]}

        warnings = []

        # Check blur (Laplacian variance) - LOWERED threshold
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        if laplacian_var < 50:  # Changed from 100 - much more lenient
            warnings.append(
                "Image appears blurry. Hold camera steady or use better focus."
            )

        # Check brightness - MORE LENIENT range
        brightness = np.mean(gray)

        if brightness < 30:  # Changed from 50 - darker OK
            warnings.append("Image is very dark. Try better lighting.")
        elif brightness > 240:  # Changed from 220 - brighter OK
            warnings.append("Image is overexposed. Reduce brightness.")

        is_good = len(warnings) == 0

        return {"is_good": is_good, "warnings": warnings}

        # Check for cage bars
        if self.detect_bars_or_cage(img):
            warnings.append(
                "Cage or fence bars detected. Please take photo without barriers between camera and pet."
            )


# Test function
def test_quality_checker():
    """Test the quality checker"""
    checker = ImageQualityChecker()

    # Test on dog image
    test_image = "data/test_images/dog.jpg"

    print(f"\nTesting quality on: {test_image}")
    result = checker.check_quality(test_image)

    print(f"Is good quality: {result['is_good']}")
    print(f"Metrics: {result['metrics']}")

    if result["warnings"]:
        print("Warnings:")
        for warning in result["warnings"]:
            print(f"  - {warning}")
    else:
        print("✓ Image quality is good!")


if __name__ == "__main__":
    test_quality_checker()
