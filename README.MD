# Face Compare Service

Face Compare Service adalah backend service berbasis **FastAPI** untuk membandingkan kemiripan wajah menggunakan **face embeddings** (ArcFace / InsightFace).

Service ini mendukung:
- Deteksi wajah
- Quality check wajah
- Perbandingan wajah (1:1)
- Registrasi wajah (enrollment)
- Identifikasi wajah terhadap database (1:N)

Project ini dirancang untuk **local development**, **proof of concept**, dan **eksperimen biometric verification**.

---

## ✨ Features

- Face detection
- Face quality check (confidence & face size)
- Face similarity comparison (1:1)
- Face enrollment (store embedding)
- Face identification (1:N)
- Cosine similarity scoring
- CPU-only (tanpa GPU)
- MySQL / MariaDB support
- Logging similarity & request tracing

---

## 🧱 Tech Stack

- Python 3.10+
- FastAPI
- InsightFace (ArcFace)
- OpenCV
- NumPy
- MySQL / MariaDB

---

## 📁 Project Structure

```text
.
├── app.py
├── services/
│   ├── face_service.py        # Face detection & embedding
│   └── face_repository.py    # Database access (MySQL/MariaDB)
├── utils/
│   └── image.py               # Image helper
├── logs/
│   └── app.log
├── requirements.txt
├── .env.example
└── README.md