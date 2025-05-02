import rclpy
from rclpy.node import Node

class GestureNode(Node):
    def __init__(self):
        super().__init__('gesture_node')
        self.get_logger().info('Nodo de gestos iniciado correctamente')

def main(args=None):
    rclpy.init(args=args)
    node = GestureNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

