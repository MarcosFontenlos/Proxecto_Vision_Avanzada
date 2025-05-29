# descargar_yolov8m.py
from ultralytics import YOLO

# Cargar o modelo YOLOv8m preentrenado (só se descarga se non está xa descargado)
modelo = YOLO('yolov8n.pt')

print("Modelo YOLOv8m descargado correctamente.")

