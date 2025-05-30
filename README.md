# 🤖 Proxecto Visión Artificial Avanzada

Este proyecto desarrolla un sistema de percepción visual modular sobre **ROS 2**, capaz de detectar personas y gestos mediante visión artificial, y controlar el movimiento de un robot móvil (Kompai) en función de la información captada.

## 🧠 Arquitectura del Sistema

El sistema está compuesto por tres nodos principales:

* 🟡 **`gesture_movimiento.py`**
  Versión inicial con detección de gestos de mano mediante MediaPipe y seguimiento de personas usando YOLOv8, además también se integrará la deteccion de botellas. Publica comandos en `/robulab10/cmd_vel`.

* 🟡 **`gesture_segmentacion.py`**
  Combina detección de gestos con MediaPipe y seguimiento visual avanzado de personas y botellas con YOLOv8 + segmentación. Usa también un sistema de cambio de estado basado en gestos y lógica de transición inteligente.

## 📁 Estructura del Workspace

```
Proxecto_Vision_Avanzada/
├── gesture_movimiento.py               # Versión inicial (YOLO + MediaPipe)
├── gesture_segmentacion.py            # Versión secundaria (YOLO + segmentación + gestos)
├── gesture_segmentacion_CamaraLocal.py
├── camara_ip.py                       # Captura por IP
├── yolov8n.pt                         # Modelo YOLOv8 Nano (personas)
├── yolov8n-seg.pt                     # Modelo YOLOv8 Nano Segmentación (botellas)
├── yolov8m.pt                         # Modelo YOLOv8 Medium (alternativo)
├── metricas.csv                       # Métricas automáticas
├── metricas_rendimiento.csv           # Métricas personalizadas
├── Dockerfile                         # Imagen base con ROS + YOLO + OpenCV
├── README.md                          # Este documento
└── Proposta_Traballo/
    ├── PropostaTaballoFM_plantilla.tex # Memoria en LaTeX
    └── PropostaTaballoFM_plantilla.pdf # PDF generado
```

## ⚙️ Instalación

1. Clonar el repositorio:

```bash
git clone https://github.com/MarcosFontenlos/Proxecto_Vision_Avanzada.git
cd Proxecto_Vision_Avanzada
```

2. Instalar dependencias (si **no** usas Docker):

```bash
pip3 install ultralytics opencv-python mediapipe
```

3. O ejecutar dentro del Docker preparado (recomendado), debes modificar ROS_IP y device /dev/video... por el numero asociado a la cámara:

```bash
xhost +SI:localuser:root && echo $DISPLAY && docker run -it --rm \
  --network=host \
  --gpus all \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -e ROS_MASTER_URI=http://192.168.1.30:11311 \
  -e ROS_IP=192.168.1.188 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v /home/alex/Desktop/Vision_Artificial_Avanzada/Proyecto/Proxecto_Vision_Avanzada:/catkin_ws/src \
  --device /dev/dri:/dev/dri \
  --device /dev/video2:/dev/video2 \
  marcosfontenlos/ros_vision_base:ws3
```

## ▶️ Ejecución

### ✅ Opcion 1: 

```bash
python3 gesture_movimiento.py --cam-index 2
```

### ✅ Opcion 2

```bash
python3 gesture_segmentacion.py --cam-index 2
```

> Puedes usar `--ip` para conectar directamente a la Axis Camera:

```bash
python3 gesture_segmentacion.py --ip
```

---

## 🕹 Conexión y Movimiento

### 🔌 Conexión:

* Pulsar el botón en la parte trasera del mando.
* Pulsar **Start**.
* Esperar a que se encienda la luz.

### 🗺 Movimiento:

* Pulsar **gatillo izquierdo (L2)**.
* Mantener presionado **botón A**.
* Usar **joystick izquierdo** para desplazarse.

## 📡 Topics utilizados

* `/robulab10/cmd_vel`  → Movimiento del robot (Twist).
* `/detecciones_yolo`   → Detecta personas/objetos (futuro).
* `/gestos_mano`        → Gestos detectados (futuro).

## 🛠 Tecnologías

* ROS Noetic
* Python 3.10
* YOLOv8 (Ultralytics)
* OpenCV
* MediaPipe 
* Comunicación HTTP/TCP con robot Kompai

## 👥 Autores

* **Marcos Fontenlos**
  [@MarcosFontenlos](https://github.com/MarcosFontenlos)

* **Alejandro Solar**
  [@AlejandroSIRob](https://github.com/AlejandroSIRob)

