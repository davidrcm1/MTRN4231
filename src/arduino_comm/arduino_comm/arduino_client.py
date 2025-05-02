# arduino client
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger

class ArduinoClientNode(Node):
    def __init__(self):
        super().__init__('arduino_client')
        self.cli = self.create_client(Trigger, 'trigger_operation')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting...')
        self.send_request()

    def send_request(self):
        req = Trigger.Request()
        self.future = self.cli.call_async(req)
        self.future.add_done_callback(self.handle_response)

    def handle_response(self, future):
        response = future.result()
        if response.success:
            self.get_logger().info(f"Response: {response.message}")
        else:
            self.get_logger().error(f"Operation failed: {response.message}")

def main(args=None):
    rclpy.init(args=args)
    node = ArduinoClientNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
