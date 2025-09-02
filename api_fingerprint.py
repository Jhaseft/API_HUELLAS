from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil, os, tempfile
import cv2
from fingerprint_processor import comparar_huellas

app = FastAPI(title="Fingerprint API")

# =========================
# CORS
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

TMP_DIR = tempfile.gettempdir()

# =========================
# Utilidades
# =========================
def save_upload_file(upload_file: UploadFile) -> str:
    """Guarda un archivo subido en un archivo temporal y devuelve su path"""
    ext = os.path.splitext(upload_file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext, dir=TMP_DIR) as tmp:
        shutil.copyfileobj(upload_file.file, tmp)
        return tmp.name

# =========================
# Endpoints
# =========================
@app.get("/")
async def root():
    return {"message": "Fingerprint API está corriendo"}

@app.post("/compare-fingerprints")
async def compare_fingerprints(
    fingerprint1: UploadFile = File(...),
    fingerprint2: UploadFile = File(...),
):
    try:
        path1 = save_upload_file(fingerprint1)
        path2 = save_upload_file(fingerprint2)

        resultado = comparar_huellas(path1, path2)

        return JSONResponse(content=resultado)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            content={"error": str(e), "mensajes": ["Ocurrió un error al procesar las huellas"]},
            status_code=500
        )
