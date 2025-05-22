FROM marcosfontenlos/ros_vision_base:ws2

# Instalar dependencias necesarias para soporte GUI de OpenCV y acceso a cámara
RUN apt update && apt install -y \
    libgl1 \
    libgtk2.0-dev \
    pkg-config \
    libsm6 \
    libxrender1 \
    libxext6 \
    libcanberra-gtk-module \
    libglib2.0-0 \
    python3-pip \
    && apt clean

# Instalar python3-tk sin confirmación interactiva
RUN apt-get update && apt-get install -y python3-tk

# Directorio de trabajo
WORKDIR /catkin_ws/src

# Comando por defecto al ejecutar el contenedor
CMD ["bash"]
