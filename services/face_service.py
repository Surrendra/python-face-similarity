import numpy as np
from insightface.app import FaceAnalysis
from fastapi import HTTPException


class FaceService:
    def __init__(self):
        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=-1, det_size=(640, 640))

    def detect_faces(self, img):
        return self.app.get(img)

    def get_best_face(self, img):
        faces = self.detect_faces(img)
        if not faces:
            raise HTTPException(status_code=422, detail="No face detected")

        return max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
        )

    def get_embedding(self, img) -> np.ndarray:
        face = self.get_best_face(img)
        return face.embedding, face

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        a = a / np.linalg.norm(a)
        b = b / np.linalg.norm(b)
        return float(np.dot(a, b))