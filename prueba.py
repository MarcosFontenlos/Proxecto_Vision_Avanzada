import cv2

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)  # Usa V4L2 en Linux

if not cap.isOpened():
    print("❌ No se pudo abrir la cámara.")
    exit()

# Verifica propiedades de la cámara
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Resolución: {width}x{height}")

cv2.namedWindow('Vista de la cámara', cv2.WINDOW_GUI_NORMAL)

while True:
    ret, frame = cap.read()
    
    if not ret or frame is None or frame.size == 0:
        print("❌ Frame inválido. Revisa la conexión de la cámara.")
        break

    cv2.imshow('Vista de la cámara', frame)
    
    if cv2.waitKey(30) == ord('q'):
        print("✅ Ventana cerrada por el usuario.")
        break

cap.release()
cv2.destroyAllWindows()
