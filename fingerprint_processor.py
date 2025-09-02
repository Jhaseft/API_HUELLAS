import cv2
import numpy as np
from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize

# =========================
# --- PREPROCESAR HUELLA ---
# =========================
def preprocess_fingerprint(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"No se pudo abrir la imagen: {img_path}")

    # Normalizar tamaño
    img = cv2.resize(img, (300, 300))

    # Mejorar contraste
    img_eq = cv2.equalizeHist(img)

    # Suavizado
    img_blur = cv2.GaussianBlur(img_eq, (3, 3), 0)

    # Binarización Otsu
    thresh = threshold_otsu(img_blur)
    binary = img_blur > thresh
    binary = (binary.astype(np.uint8)) * 255

    # Esqueleto
    skeleton = skeletonize(binary // 255)
    skeleton = (skeleton.astype(np.uint8)) * 255

    return skeleton

# =========================
# --- COMPARAR HUELLAS ---
# =========================
def compare_fingerprints(img1, img2, method="ORB"):
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    if method == "ORB":
        detector = cv2.ORB_create(nfeatures=1500)
        norm_type = cv2.NORM_HAMMING
    else:
        detector = cv2.SIFT_create()
        norm_type = cv2.NORM_L2

    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return 0.0

    # Matching con ratio test
    bf = cv2.BFMatcher(norm_type)
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

    if len(good_matches) < 4:
        return 0.0

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if mask is None:
        return 0.0

    inliers = mask.ravel().sum()
    score = inliers / max(len(kp1), len(kp2))
    return score

# =========================
# --- FUNCIÓN PRINCIPAL ---
# =========================
def comparar_huellas(path1, path2):
    img1 = preprocess_fingerprint(path1)
    img2 = preprocess_fingerprint(path2)

    score = compare_fingerprints(img1, img2)

    return {
        "similarity": float(score),
        "resultado": "✅ Coinciden" if score > 0.20 else "❌ No coinciden"
    }

