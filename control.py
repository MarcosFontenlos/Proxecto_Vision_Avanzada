#!/usr/bin/env python3

import cv2
import mediapipe as mp
import sys
import rospy
from std_msgs.msg import Bool


class GestureDetector:
    def __init__(self, cam_index=0):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False,
                                         max_num_hands=1,
                                         min_detection_confidence=0.7)
        self.mp_drawing = mp.solutions.drawing_utils
        self.cap = cv2.VideoCapture(cam_index)

        self.estado_anterior = None
        self.contador_modo_automatico = 0
        self.modo_automatico_activado = False
        self.contador_parar_en_modo_auto = 0

        # Inicializar ROS y publisher
        rospy.init_node('gesture_control_node')
        self.pub_modo_botella = rospy.Publisher('/modo_botella', Bool, queue_size=1)

        if not self.cap.isOpened():
            print(f"[ERROR] No se pudo abrir la cámara con índice {cam_index}")
            exit(1)
        else:
            print(f"[INFO] Cámara abierta correctamente (índice {cam_index})")

    def detectar_dedos(self, hand_landmarks):
        dedos = []
        dedos.append(hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x)
        dedos.append(hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y)
        dedos.append(hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y)
        dedos.append(hand_landmarks.landmark[16].y < hand_landmarks.landmark[14].y)
        dedos.append(hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y)
        return dedos

    def detectar_estado(self, hand_landmarks):
        dedos = self.detectar_dedos(hand_landmarks)
        total_dedos = dedos.count(True)
        if total_dedos == 5:
            return "PARAR"
        elif total_dedos == 1 and dedos[1]:
            return "GIRAR"
        elif total_dedos == 0:
            return "MODO AUTOMATICO"
        elif total_dedos == 2:
            return "IZQUIERDA"
        elif total_dedos == 3:
            return "DERECHA"
        else:
            return "NINGUNO"

    def modo_automatico(self, frame):
        overlay = frame.copy()
        alpha = 0.5
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
        frame_con_texto = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        cv2.putText(frame_con_texto, "MODO AUTOMATICO ACTIVADO", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame_con_texto, "No acepta ordenes", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame_con_texto, "PARAR (5 dedos) para salir", (10, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return frame_con_texto

    def run(self):
        cv2.namedWindow('Detección de Gestos', cv2.WINDOW_NORMAL)
        rate = rospy.Rate(10)

        while not rospy.is_shutdown():
            ret, frame = self.cap.read()
            if not ret:
                print("[WARN] No se pudo leer de la cámara")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)

            estado = "NINGUNO"
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    estado = self.detectar_estado(hand_landmarks)
                    self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

            # Modo automático
            if estado == "MODO AUTOMATICO":
                self.contador_modo_automatico += 1
                if self.contador_modo_automatico >= 10 and not self.modo_automatico_activado:
                    self.modo_automatico_activado = True
                    print("[INFO] Modo automático activado")
            else:
                self.contador_modo_automatico = 0

            if self.modo_automatico_activado:
                if estado == "PARAR":
                    self.contador_parar_en_modo_auto += 1
                    if self.contador_parar_en_modo_auto >= 10:
                        self.modo_automatico_activado = False
                        self.contador_parar_en_modo_auto = 0
                        print("[INFO] Modo automático desactivado tras 10 frames de PARAR")
                else:
                    self.contador_parar_en_modo_auto = 0

            # Publicar estado a /modo_botella
            self.pub_modo_botella.publish(Bool(data=self.modo_automatico_activado))

            if estado != self.estado_anterior and not self.modo_automatico_activado:
                print(f"[DEBUG] Cambio de estado: {self.estado_anterior} -> {estado}")
                self.estado_anterior = estado

            if self.modo_automatico_activado:
                frame = self.modo_automatico(frame)
            else:
                cv2.putText(frame, estado, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Detección de Gestos', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            rate.sleep()

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    cam_index = 0
    if len(sys.argv) > 1:
        try:
            opcion = int(sys.argv[1])
            cam_index = 1 if opcion == 1 else 0
        except ValueError:
            print("Uso: python3 gesture_node.py [opcion]")
            print("Opcion 1: cámara USB (TurtleBot), Opcion 2: webcam (por defecto)")
    detector = GestureDetector(cam_index)
    detector.run()
