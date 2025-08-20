import cv2
import numpy as np

# Ruta de la imagen de referencia
imagen_ref_path = r"c:\Users\cuarto_4c\Documents\botella\FANTA-250CC-DESECHABLE.jpg"
img_ref = cv2.imread(imagen_ref_path, 0)  # Leer en escala de grises

# Verificar que la imagen se cargó
if img_ref is None:
    raise FileNotFoundError(f"No se encontró la imagen en: {imagen_ref_path}")

# Crear detector ORB
orb = cv2.ORB_create(nfeatures=1000)
kp_ref, des_ref = orb.detectAndCompute(img_ref, None)

# Configurar FLANN para ORB (usa LSH)
FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH,
                    table_number=6,      # 12 en algunos casos
                    key_size=12,         # 20 en algunos casos
                    multi_probe_level=1) # 2 en algunos casos
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Inicializar cámara (usa 0 si la cámara principal)
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        print("No se pudo acceder a la cámara")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp_frame, des_frame = orb.detectAndCompute(gray, None)
    
    detected = False
    if des_ref is not None and des_frame is not None:
        if len(des_ref) >= 2 and len(des_frame) >= 2:  # Evitar error de FLANN
            matches = flann.knnMatch(des_ref, des_frame, k=2)

            # Ratio Test de Lowe con validación
            good_matches = []
            for match in matches:
                if len(match) == 2:  # Solo si hay dos vecinos
                    m, n = match
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)

            if len(good_matches) > 15:  # Ajusta este número según pruebas
                detected = True
                cv2.putText(frame, "Botella Fanta Detectada", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if not detected:
        cv2.putText(frame, "No se detecta botella", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Reconocimiento Botella", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
