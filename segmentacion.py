from ultralytics import YOLO
import cv2

# === 1. Cargar modelo YOLOv8 de segmentación ===
modelo = YOLO('yolov8n-seg.pt')  # Asegúrate de que el modelo esté en la ruta correcta

# === 2. Abrir webcam (0 = cámara por defecto) ===
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("No se pudo abrir la cámara")
    exit()

print("Presiona 'q' para salir")

# Nombre de la ventana (debe ser el mismo en cada cv2.imshow)
window_name = "Segmentación YOLOv8 (Webcam)"
cv2.namedWindow(window_name)  # Opcional: ayuda a controlar el tamaño de la ventana

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # === 3. Hacer predicción con el modelo ===
    resultados = modelo(frame)

    # === 4. Dibujar resultados de segmentación ===
    frame_segmentado = resultados[0].plot()
    
    # === 5. Mostrar en la misma ventana (sin abrir nuevas) ===
    cv2.imshow(window_name, frame_segmentado)
    
    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === 6. Liberar recursos ===
cap.release()
cv2.destroyAllWindows()  # Cierra todas las ventanas al finalizar
