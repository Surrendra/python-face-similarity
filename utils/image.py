import cv2
import numpy as np
from fastapi import UploadFile, HTTPException


def read_image(upload: UploadFile) -> np.ndarray:
    content = upload.file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty image file")

    nparr = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image format")

    return img