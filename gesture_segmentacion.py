#!/usr/bin/env python3
import cv2
import mediapipe as mp
import rospy
import math
import time
import numpy as np
from geometry_msgs.msg import Twist
from ultralytics import YOLO

class GestureSegmentDetector:
    def __init__(self, cam_index=0):
        # Inicializar MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False,
                                         max_num_hands=1,
                                         min_detection_confidence=0.7)
        self.mp_drawing = mp.solutions.drawing_utils

        # Inicializar YOLOv8 para segmentación
        self.modelo_segmentacion = YOLO('yolov8n-seg.pt')  # Modelo de segmentación

        # Inicializar cámara
        self.cap = cv2.VideoCapture(cam_index)
        if not self.cap.isOpened():
            print("[ERROR] No se pudo abrir la cámara")
            exit()

        # Inicializar ROS (opcional, solo si necesitas controlar un robot)
        try:
            rospy.init_node('gesture_segment_controller', anonymous=True)
            self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        except:
            print("[INFO] ROS no inicializado. Modo sin control de robot.")

        # Estados y contadores
        self.estado_actual = "RECONOCIMIENTO"
        self.gesto_actual = "NINGUNO"
        self.mano_centrada_flag = False

        # Configuración de ventana
        self.window_name = "Gesture + Segmentación"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def analizar_gesto(self, hand_landmarks):
        # Detectar dedos levantados (0 = cerrado, 5 = abierto)
        dedos = [
            hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x,  # Pulgar
            hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y,  # Índice
            hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y,  # Medio
            hand_landmarks.landmark[16].y < hand_landmarks.landmark[14].y,  # Anular
            hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y,  # Meñique
        ]
        return dedos.count(True)

    def mano_centrada(self, hand_landmarks, frame_width):
        mano_x = hand_landmarks.landmark[9].x * frame_width  # Punto medio de la palma
        margen = frame_width * 0.2
        return abs(mano_x - frame_width/2) < margen

    def run(self):
        while not rospy.is_shutdown():
            ret, frame = self.cap.read()
            if not ret:
                continue

            # Procesamiento de gestos
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results_hands = self.hands.process(frame_rgb)

            self.gesto_actual = "NINGUNO"
            self.mano_centrada_flag = False

            if results_hands.multi_hand_landmarks:
                for hand_landmarks in results_hands.multi_hand_landmarks:
                    dedos_levantados = self.analizar_gesto(hand_landmarks)
                    self.mano_centrada_flag = self.mano_centrada(hand_landmarks, frame.shape[1])
                    self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                    # Clasificar gesto
                    if dedos_levantados == 5:
                        self.gesto_actual = "PARAR"
                    elif dedos_levantados == 0:
                        self.gesto_actual = "CERRADO"
                    elif 1 <= dedos_levantados <= 2:
                        self.gesto_actual = "OBJETO"
                    else:
                        self.gesto_actual = "OTRO"

            # Segmentación con YOLOv8
            resultados_seg = self.modelo_segmentacion(frame)
            frame_segmentado = resultados_seg[0].plot()  # Frame con máscaras de segmentación

            # Mostrar información
            cv2.putText(frame_segmentado, f"Estado: {self.estado_actual}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame_segmentado, f"Gesto: {self.gesto_actual}", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame_segmentado, f"Mano Centrada: {'SI' if self.mano_centrada_flag else 'NO'}", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Control del robot (ejemplo básico)
            twist = Twist()
            if self.gesto_actual == "PARAR" and self.mano_centrada_flag:
                twist.linear.x = 0
                twist.angular.z = 0
                self.estado_actual = "PARAR"
            elif self.gesto_actual == "OBJETO" and self.mano_centrada_flag:
                # Aquí podrías añadir lógica para seguir objetos segmentados
                twist.linear.x = 0.1
                twist.angular.z = 0
                self.estado_actual = "SEGUIMIENTO_OBJETO"
            else:
                twist.linear.x = 0
                twist.angular.z = 0
                self.estado_actual = "RECONOCIMIENTO"

            try:
                self.pub.publish(twist)
            except:
                pass

            # Mostrar frame combinado
            cv2.imshow(self.window_name, frame_segmentado)

            # Salir con ESC
            if cv2.waitKey(1) & 0xFF == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = GestureSegmentDetector()
    detector.run()
