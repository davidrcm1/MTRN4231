import rclpy
from rclpy.node import Node

from key_pos_msgs.msg import KeyPositionArray, KeyPosition, MoveItCoords

Z_GLOBAL = 20
QX_GLOBAL = 0
QY_GLOBAL = 0
QZ_GLOBAL = 0
QW_GLOBAL = 1

class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.moveit_publisher = self.create_publisher(MoveItCoords, 'moveit_target', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        # Create and publish the MoveitCoords message
        target_msg = MoveItCoords()
        target_msg.x = float(20)
        target_msg.y = float(20)
        target_msg.z = float(Z_GLOBAL)
        target_msg.qx = float(QX_GLOBAL)
        target_msg.qy = float(QY_GLOBAL)
        target_msg.qz = float(QZ_GLOBAL)
        target_msg.qw = float(QW_GLOBAL)

        self.moveit_publisher.publish(target_msg)
        self.get_logger().info(f"Published MoveIt target")
        


def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher()

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()