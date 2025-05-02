import rclpy
from rclpy.node import Node

from key_pos_msgs.msg import KeyPositionArray, KeyPosition

Z_GLOBAL = 20
QX_GLOBAL = 0
QY_GLOBAL = 0
QZ_GLOBAL = 0
QW_GLOBAL = 1

class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher = self.create_publisher(KeyPositionArray, 'key_positions', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        # Create and publish the MoveitCoords message
        letters = ["R1", "O1", "Y1", "L1", "G1", "B1", "N1", "P1", "R2", "O2", "Y2", "L2", "G2", "B2", "N2"]
        key_position_array = KeyPositionArray()
        for letter in letters:
            key_pos = KeyPosition()
            key_pos.letter = letter
            key_pos.x = 250
            key_pos.y = 250
            key_position_array.key_positions.append(key_pos)

        key_position_array.people_detected = False
        key_position_array.dots_detected = True
        # Publish the array of key positions
        self.publisher.publish(key_position_array)
        self.get_logger().info(f"Published key positions: {[f'{kp.letter}: ({kp.x},{kp.y})' for kp in key_position_array.key_positions]}, person detected: {key_position_array.people_detected}, dots detected: {key_position_array.dots_detected}")
                
        


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