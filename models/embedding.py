"""
Embedding module - generates feature vectors for pet faces and bodies
"""

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import models
from typing import Optional, Tuple
from pathlib import Path
import onnxruntime as ort


class PetEmbedder:
    """Generates embeddings for pet faces and bodies"""

    def __init__(self, device: str = "cpu"):
        """
        Initialize embedding models

        Args:
            device: 'cpu' or 'cuda'
        """
        self.device = device
        print(f"Initializing embedders on {device}")

        # Body embeddings: ResNet50 (pretrained on ImageNet)
        print("Loading ResNet50 for body embeddings...")
        self.body_model = models.resnet50(pretrained=True)
        # Remove classification layer, keep feature extractor
        self.body_model = torch.nn.Sequential(*list(self.body_model.children())[:-1])
        self.body_model.eval()
        self.body_model.to(device)

        # Face embeddings: We'll use InsightFace ArcFace (load later with ONNX)
        # For MVP, we'll use ResNet for both - you can add ArcFace in Day 2-3
        self.face_model = None  # Placeholder for now

        # Image preprocessing
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def generate_body_embedding(self, image: np.ndarray) -> np.ndarray:
        """
        Generate 2048-d embedding for pet body

        Args:
            image: OpenCV image (numpy array, BGR format)

        Returns:
            2048-d feature vector
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Preprocess
        img_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)

        # Generate embedding
        with torch.no_grad():
            embedding = self.body_model(img_tensor)

        # Flatten and convert to numpy
        embedding = embedding.squeeze().cpu().numpy()

        # Normalize (L2 norm)
        embedding = embedding / np.linalg.norm(embedding)

        return embedding

    def generate_face_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Generate 512-d embedding for pet face (placeholder for now)

        Args:
            image: OpenCV image (numpy array, BGR format)

        Returns:
            512-d feature vector or None if face model not loaded
        """
        if self.face_model is None:
            # For MVP, use body model for faces too
            # Will add proper ArcFace model later
            body_emb = self.generate_body_embedding(image)
            # Reduce dimensionality for consistency (2048 -> 512)
            return body_emb[:512]

        # TODO: Add ArcFace implementation
        return None

    def generate_embeddings(self, image: np.ndarray, mode: str = "body") -> np.ndarray:
        """
        Generate embeddings based on mode

        Args:
            image: OpenCV image
            mode: 'body', 'face', or 'both'

        Returns:
            Feature vector (512-d for face, 2048-d for body)
        """
        if mode == "body":
            return self.generate_body_embedding(image)
        elif mode == "face":
            return self.generate_face_embedding(image)
        elif mode == "both":
            body_emb = self.generate_body_embedding(image)
            face_emb = self.generate_face_embedding(image)
            # Concatenate
            return np.concatenate([body_emb, face_emb])
        else:
            raise ValueError(f"Invalid mode: {mode}")

    @staticmethod
    def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings

        Args:
            emb1: First embedding
            emb2: Second embedding

        Returns:
            Similarity score (0-1)
        """
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))


# Test function
def test_embedder():
    """Test embedding generation"""
    from models.detection import PetDetector

    # Initialize
    detector = PetDetector()
    embedder = PetEmbedder()

    # Test image
    test_image = "data/test_images/dog.jpg"

    if not Path(test_image).exists():
        print(f"Test image not found: {test_image}")
        return

    print(f"\nTesting embedding generation on: {test_image}")

    # Detect and crop
    detections, cropped = detector.detect_and_crop(test_image)

    if len(cropped) == 0:
        print("No pets detected!")
        return

    print(f"Found {len(cropped)} pet(s)")

    # Generate embeddings
    for i, crop in enumerate(cropped):
        print(f"\nPet {i+1}:")

        # Body embedding
        body_emb = embedder.generate_body_embedding(crop)
        print(f"  Body embedding shape: {body_emb.shape}")
        print(f"  Body embedding (first 5): {body_emb[:5]}")

        # Face embedding (using body model for now)
        face_emb = embedder.generate_face_embedding(crop)
        print(f"  Face embedding shape: {face_emb.shape}")
        print(f"  Face embedding (first 5): {face_emb[:5]}")

        # Test similarity with itself (should be ~1.0)
        similarity = embedder.cosine_similarity(body_emb, body_emb)
        print(f"  Self-similarity: {similarity:.4f} (should be ~1.0)")

    # Test with two different images (if you have them)
    test_image2 = "data/test_images/cropped_0.jpg"
    if Path(test_image2).exists():
        print(f"\n\nComparing with cropped image:")
        crop2 = cv2.imread(test_image2)
        emb1 = embedder.generate_body_embedding(cropped[0])
        emb2 = embedder.generate_body_embedding(crop2)
        similarity = embedder.cosine_similarity(emb1, emb2)
        print(f"  Similarity: {similarity:.4f}")


if __name__ == "__main__":
    test_embedder()
