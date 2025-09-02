import cv2
from fingerprint import preprocess_fingerprint, compare_fingerprints

if __name__ == "__main__":
    # Ruta de las dos huellas a comparar
    huella1_path = "data/persona1.jpg"
    huella2_path = "data/persona1.jpg"

    # Preprocesar ambas
    img1 = preprocess_fingerprint(huella1_path, save_steps=True, prefix="data/huella1_debug")
    img2 = preprocess_fingerprint(huella2_path, save_steps=True, prefix="data/huella2_debug")

    # Guardar procesadas
    cv2.imwrite("database/huella1_processed.png", img1)
    cv2.imwrite("database/huella2_processed.png", img2)

    # Comparar solo esas dos
    score = compare_fingerprints(img1, img2, method="ORB")

    print(f"Comparación entre {huella1_path} y {huella2_path}")
    print(f"Score de similitud: {score:.2f}")

    if score >= 0.70:
        print("✅ Coinciden")
    else:
        print("❌ No coinciden")
