import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/alex/Desktop/Visión Artificial Avanzada/Proyecto/Proxecto_Vision_Avanzada/vision_ws/install/hand_gesture'
