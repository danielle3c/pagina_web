import cv2
import numpy as np

# ====== Lista de imágenes de referencia ======
imagenes_ref = [
    ("C:/Users/sofid/Documents/botella/FANTA-250CC-DESECHABLE.jpg", "Botella"),
    ("C:/Users/sofid/Documents/botella/descarga.jpeg", "Cartón"),
    (r"c:\Users\sofid\Documents\botella\d500351d-23dd-48fb-98c4-1964390cd8bc-lg.jpg", "Lata")  # Ruta corregida
]

# ====== Cargar imágenes de referencia ======
orb = cv2.ORB_create(nfeatures=1000)
referencias = []

for ruta, nombre in imagenes_ref:
    img = cv2.imread(ruta, 0)  # Cargar en escala de grises
    if img is None:
        print(f"⚠ No se encontró la imagen: {ruta}")
        continue
    kp, des = orb.detectAndCompute(img, None)
    referencias.append((kp, des, nombre))

if not referencias:
    raise FileNotFoundError("No se pudieron cargar las imágenes de referencia.")

# ====== Configurar FLANN ======
FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH,
                    table_number=6,
                    key_size=12,
                    multi_probe_level=1)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# ====== Inicializar cámara ======
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # Cambia a 0 si es la webcam principal

while True:
    ret, frame = cap.read()
    if not ret:
        print("No se pudo acceder a la cámara")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp_frame, des_frame = orb.detectAndCompute(gray, None)

    objeto_detectado = "Nada detectado"

    if des_frame is not None and len(des_frame) >= 2:
        for kp_ref, des_ref, nombre in referencias:
            if des_ref is not None and len(des_ref) >= 2:
                matches = flann.knnMatch(des_ref, des_frame, k=2)

                # Ratio Test de Lowe con control de errores
                good_matches = []
                for match in matches:
                    if len(match) == 2:
                        m, n = match
                        if m.distance < 0.7 * n.distance:
                            good_matches.append(m)

                if len(good_matches) > 15:
                    objeto_detectado = f"{nombre} detectado"
                    break

    cv2.putText(frame, objeto_detectado, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0) if "detectado" in objeto_detectado else (0, 0, 255),
                2)

    cv2.imshow("Reconocimiento de Objetos", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
