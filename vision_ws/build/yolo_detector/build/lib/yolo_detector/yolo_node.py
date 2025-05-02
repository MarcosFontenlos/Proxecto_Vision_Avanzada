import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO

class YoloDetector(Node):
    def __init__(self):
        super().__init__('yolo_node')
        self.bridge = CvBridge()
        self.model = YOLO('yolov8n.pt')
        self.cap = cv2.VideoCapture(0)
        self.timer = self.create_timer(0.1, self.detect_callback)
        cv2.namedWindow('Detección YOLO', cv2.WINDOW_NORMAL)  # Crea la ventana una vez al inicio

    def detect_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn('Non se puido ler da cámara')
            return

        results = self.model(frame)[0]
        for box in results.boxes:
            cls = int(box.cls[0])
            conf = box.conf[0]
            label = self.model.names[cls]
            if label in ['person', 'hand']:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Actualiza la ventana existente en lugar de crear una nueva
        cv2.imshow('Detección YOLO', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.cap.release()
            cv2.destroyAllWindows()
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = YoloDetector()
    rclpy.spin(node)
    node.destroy_node()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()