FROM marcosfontenlos/ros_vision:latest

# Instalar dependencias para CUDA
RUN apt-get update && apt-get install -y wget gnupg2 lsb-release

# (Opcional) Instalar claves e repos de NVIDIA se necesitas compilar algo especial
# Pero para PyTorch, o máis sinxelo é instalar directamente desde pip

# Instalar PyTorch con soporte CUDA (usa unha versión que che funcione ->>)
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 

# Verifica que torch detecta CUDA (opcional)
CMD ["python3", "-c", "import torch; print(torch.cuda.is_available())"]
