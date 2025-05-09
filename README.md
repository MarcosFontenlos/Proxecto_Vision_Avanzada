
# 🤖 Proxecto Visión Artificial Avanzada

Este proyecto desarrolla un sistema de percepción visual modular sobre **ROS 2**, capaz de detectar personas y gestos mediante visión artificial, y controlar el movimiento de un robot móvil (Kompai) en función de la información captada.

## 🧠 Arquitectura del Sistema

El sistema está compuesto por tres nodos principales:

- 🟡 **`yolo_detector`**  
  Detecta personas y objetos utilizando **YOLOv8**. Publica las detecciones en el topic `/detecciones_yolo`.

- 🟢 **`hand_gesture`**  
  Detecta gestos de mano a partir de la imagen de la cámara. Publica los resultados en `/gestos_mano`.

- 🔵 **`robot_controller`**  
  Recibe los datos anteriores, decide la acción a realizar, y envía comandos al robot **Kompai** vía HTTP o sockets.

## 📁 Estructura del Workspace

```
vision_ws/
├── src/
│   ├── yolo_detector/        # Nodo YOLOv8
│   ├── hand_gesture/         # Nodo de gestos de mano
│   └── robot_controller/     # Nodo de control (por implementar)
├── yolov8n.pt                # Modelo preentrenado YOLOv8 Nano
├── install/
├── build/
└── log/
```

## ⚙️ Instalación

1. Clonar el repositorio:

```bash
git clone https://github.com/MarcosFontenlos/Proxecto_Vision_Avanzada.git
cd Proxecto_Vision_Avanzada/vision_ws
```

2. Instalar dependencias (fuera del entorno virtual si usas ROS puro):

```bash
pip3 install ultralytics opencv-python
```

3. Compilar el workspace:

```bash
colcon build
source install/setup.bash
```

## ▶️ Ejecución

### Nodo YOLO

```bash
ros2 run yolo_detector yolo_node
```

### Nodo Gestos (cuando esté implementado)

```bash
ros2 run hand_gesture gesture_node
```

### Nodo Controlador del Robot (futuro)

```bash
ros2 run robot_controller controller_node
```

## 🕹 Conexión y Movimiento
## 🔌 Conexión:
 
- Pulsar el botón en la parte trasera del mando.  
- Pulsar el botón **Start**.
- Encenderá la luz. 

## 🧭 Movimiento:

- Pulsar el **gatillo izquierdo**.  
- Pulsar el **botón A**.  
- Usar el **joystick izquierdo** para moverse.
    
## 📡 Topics utilizados

- `/detecciones_yolo` – Detecciones de objetos con YOLO.
- `/gestos_mano` – Gestos detectados con visión.
- `/comando_robot` – Comandos al robot según percepción.

## 🛠 Tecnologías

- ROS 2 Humble
- Python 3.10
- YOLOv8 (Ultralytics)
- OpenCV
- Comunicación HTTP/TCP con robot Kompai

## 👥 Autores

- **Marcos Fontenlos**  
  [@MarcosFontenlos](https://github.com/MarcosFontenlos)

- **Alejandro Solar**  
  [@AlejandroSIRob](https://github.com/AlejandroSIRob)
