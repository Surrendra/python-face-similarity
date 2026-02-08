import uuid
import logging
from fastapi import FastAPI, UploadFile, File, Request
from utils.image import read_image
from services.face_service import FaceService

logging.basicConfig(
    filename="logs/app.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

app = FastAPI(title="Face Compare Service")
face_service = FaceService()


@app.post("/face/compare")
async def compare_faces(
    request: Request,
    image_1: UploadFile = File(...),
    image_2: UploadFile = File(...),
    threshold: float = 0.8
):
    request_id = str(uuid.uuid4())

    img1 = read_image(image_1)
    img2 = read_image(image_2)

    emb1, face1 = face_service.get_embedding(img1)
    emb2, face2 = face_service.get_embedding(img2)

    similarity = face_service.cosine_similarity(emb1, emb2)
    match = similarity >= threshold

    logger.info(
        f"[{request_id}] similarity={similarity:.4f} match={match}"
    )

    return {
        "request_id": request_id,
        "similarity": similarity,
        "match": match,
        "threshold": threshold,
    }

@app.post("/face/detect")
async def detect_face(
    image: UploadFile = File(...)
):
    img = read_image(image)
    faces = face_service.detect_faces(img)

    return {
        "faces_detected": len(faces),
        "faces": [
            {
                "bbox": face.bbox.tolist(),
                "det_score": float(face.det_score)
            }
            for face in faces
        ]
    }

@app.post("/face/quality-check")
async def face_quality_check(
    image: UploadFile = File(...)
):
    img = read_image(image)
    face = face_service.get_best_face(img)

    x1, y1, x2, y2 = face.bbox
    face_width = x2 - x1
    face_height = y2 - y1

    quality = {
        "det_score": float(face.det_score),
        "face_width": int(face_width),
        "face_height": int(face_height),
        "is_good": (
            face.det_score >= 0.9 and
            face_width >= 80 and
            face_height >= 80
        )
    }

    return quality