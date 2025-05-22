FROM marcosfontenlos/ros_vision_base:cuda124


# Instalamos herramientas necesarias para catkin
RUN apt-get update && apt-get install -y \
    python3-catkin-tools \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Creamos el workspace catkin
RUN mkdir -p /catkin_ws/src

# Definimos el directorio de trabajo
WORKDIR /catkin_ws

# Inicializamos el workspace (catkin_make o catkin build)
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && catkin_make"

# Fuente automática del setup.bash del workspace para cuando abras el contenedor
RUN echo "source /catkin_ws/devel/setup.bash" >> /root/.bashrc

# Comando por defecto al ejecutar el contenedor
CMD ["bash"]

