import cv2
import numpy as np
from fingerprint_processor import preprocess_fingerprint, compare_fingerprints

# =========================
# Funciones de variación artificial
# =========================
def rotate_image(img, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=255)

def add_noise(img, mean=0, sigma=15):
    noise = np.random.normal(mean, sigma, img.shape).astype(np.uint8)
    noisy = cv2.add(img, noise)
    return noisy

def crop_center(img, crop_ratio=0.85):
    h, w = img.shape[:2]
    ch, cw = int(h * crop_ratio), int(w * crop_ratio)
    start_y, start_x = (h - ch) // 2, (w - cw) // 2
    return img[start_y:start_y + ch, start_x:start_x + cw]

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    huella1_path = "data/persona5.jpg"
    huella2_path = "data/persona4.jpg"

    # Procesar huellas originales
    img1 = preprocess_fingerprint(huella1_path)
    img2 = preprocess_fingerprint(huella2_path)

    cv2.imwrite("database/huella1_processed.png", img1)
    cv2.imwrite("database/huella2_processed.png", img2)

    # Comparar huellas distintas
    score = compare_fingerprints(img1, img2, method="ORB")
    print(f"\nComparación entre {huella1_path} y {huella2_path}")
    print(f"Score de similitud: {score:.2f}")
    print("✅ Coinciden" if score >= 0.50 else "❌ No coinciden")

    # =========================
    # PRUEBAS DE VARIACIONES
    # =========================
    print("\n=== PRUEBAS CON LA MISMA HUELLA (persona4) PERO MODIFICADA ===")
    variaciones = {
        "Rotada +10°": rotate_image(img1, 10),
        "Rotada -15°": rotate_image(img1, -15),
        "Con ruido": add_noise(img1),
        "Recortada": crop_center(img1)
    }

    for nombre, var_img in variaciones.items():
        score_var = compare_fingerprints(img1, var_img)
        print(f"{nombre}: Score = {score_var:.2f} → {'✅ Coinciden' if score_var >= 0.15 else '❌ No coinciden'}")
