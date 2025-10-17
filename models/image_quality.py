"""
Image quality module - checks if images are suitable for pet detection
"""

import cv2
import numpy as np
from typing import Dict, Tuple


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

    def check_quality(self, image_path: str) -> Dict:
        """
        Comprehensive quality check

        Args:
            image_path: Path to image file

        Returns:
            Dict with quality results and warnings
        """
        # Read image
        image = cv2.imread(image_path)

        if image is None:
            return {
                "is_good": False,
                "warnings": ["Could not read image file"],
                "metrics": {},
            }

        # Run checks
        brightness_ok, brightness = self.check_brightness(image)
        sharpness_ok, sharpness = self.check_sharpness(image)
        resolution_ok, (height, width) = self.check_resolution(image)

        # Collect warnings
        warnings = []

        if not brightness_ok:
            if brightness < self.min_brightness:
                warnings.append(
                    "Image is too dark. Try taking photo in better lighting."
                )
            else:
                warnings.append("Image is overexposed. Try reducing brightness.")

        if not sharpness_ok:
            warnings.append(
                "Image appears blurry. Hold camera steady or use better focus."
            )

        if not resolution_ok:
            warnings.append(
                f"Image resolution is too low ({width}x{height}). Use higher quality camera."
            )

        return {
            "is_good": brightness_ok and sharpness_ok and resolution_ok,
            "warnings": warnings,
            "metrics": {
                "brightness": brightness,
                "sharpness": sharpness,
                "resolution": (width, height),
            },
        }


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
