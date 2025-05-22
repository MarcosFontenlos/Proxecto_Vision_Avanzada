#!/usr/bin/env python

import rospy
from std_msgs.msg import String

def leer_comando(filepath):
    try:
        with open(filepath, 'r') as f:
            return f.read().strip()
    except:
        return ""

def main():
    rospy.init_node('publicador_comandos')
    pub = rospy.Publisher('/comando_robot', String, queue_size=10)
    rate = rospy.Rate(10)  # 10 Hz

    archivo = "/catkin_ws/comandos/comando.txt"  # Ruta montada desde el host

    while not rospy.is_shutdown():
        comando = leer_comando(archivo)
        if comando:
            rospy.loginfo(f"Enviando comando: {comando}")
            pub.publish(comando)
        rate.sleep()

if __name__ == '__main__':
    main()
