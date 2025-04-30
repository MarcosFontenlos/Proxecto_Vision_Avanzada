import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/marcos/USC/Vision_avanzada/Proxecto_Final/vision_ws/install/hand_gesture'
