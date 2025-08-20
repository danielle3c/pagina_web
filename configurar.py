import cv2
import numpy as np
import os

# ====== Función para cargar imágenes de una carpeta ======
def cargar_imagenes(carpeta):
    imagenes = []
    for archivo in os.listdir(carpeta):
        ruta = os.path.join(carpeta, archivo)
        img = cv2.imread(ruta, 0)  # Cargar en escala de grises
        if img is not None:
            imagenes.append(img)
    return imagenes

# ====== Cargar imágenes de referencia por categoría ======
carpeta_botellas = r"C:\Users\programacion 4C 2025\Documents\botella"
carpeta_latas = r"C:\Users\programacion 4C 2025\Documents\latas"
carpeta_cajas = r"C:\Users\programacion 4C 2025\Documents\jugo"

ref_botellas = cargar_imagenes(carpeta_botellas)
ref_latas = cargar_imagenes(carpeta_latas)
ref_cajas = cargar_imagenes(carpeta_cajas)

# ====== Crear ORB ======
orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Precomputar descriptores para cada categoría
def compute_descriptors(lista_imagenes):
    descriptores = []
    for img in lista_imagenes:
        kp, des = orb.detectAndCompute(img, None)
        if des is not None:
            descriptores.append(des)
    return descriptores

des_botellas = compute_descriptors(ref_botellas)
des_latas = compute_descriptors(ref_latas)
des_cajas = compute_descriptors(ref_cajas)

# ====== Abrir cámara ======
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp_frame, des_frame = orb.detectAndCompute(gray, None)
    category = "Desconocido"

    if des_frame is not None:
        # ====== Comparar ORB para cajas ======
        matches_caja = [bf.match(des_c, des_frame) for des_c in des_cajas]
        max_matches_caja = max([len(m) for m in matches_caja], default=0)

        # ====== Detectar botellas y latas por color ======
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Color naranja/amarillo (botellas)
        lower_orange = np.array([10,100,100])
        upper_orange = np.array([25,255,255])
        mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)

        # Color gris/plateado (latas)
        lower_gray = np.array([0,0,100])
        upper_gray = np.array([180,50,255])
        mask_gray = cv2.inRange(hsv, lower_gray, upper_gray)

        # ====== Asignar categoría ======
        if np.sum(mask_orange) > 5000:
            category = "Botella"
        elif np.sum(mask_gray) > 5000:
            category = "Lata"
        else:
            max_matches = max_matches_caja
            if max_matches > 8:
                category = "Caja de juego"

    # Mostrar resultado en la cámara
    cv2.putText(frame, category, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Clasificador", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC para salir
        break

cap.release()
cv2.destroyAllWindows()
