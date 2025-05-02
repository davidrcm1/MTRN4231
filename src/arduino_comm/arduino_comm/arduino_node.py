#arduino node
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger  # Standard service type for simple requests
import serial
import time

class ArduinoNode(Node):
    def __init__(self):
        super().__init__('util_arduino_serial')

        # Service server to handle requests
        self.service = self.create_service(Trigger, 'trigger_operation', self.handle_trigger_request)
        
        # Initialize serial communication with Arduino
        self.serial_port = None
        while self.serial_port is None:
            try:
                self.serial_port = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
                self.get_logger().info("Connected to Arduino on /dev/ttyACM0")
            except serial.SerialException as e:
                self.get_logger().error(f"Could not open serial port: {e}. Retrying in 2 seconds...")
                time.sleep(2)

        time.sleep(2)  # Wait for serial connection to stabilize

    def handle_trigger_request(self, request, response):
        """Service handler to start the operation on Arduino and await feedback."""
        try:
            # Send the operation command to Arduino
            command = "start_operation\n"
            self.serial_port.write(command.encode('utf-8'))
            self.get_logger().info("Sent to Arduino: start_operation")
            
            # Wait for feedback from Arduino
            feedback = self.wait_for_feedback()
            if feedback == "Operation complete":
                response.success = True
                response.message = "Operation completed successfully on Arduino."
            else:
                response.success = False
                response.message = "Failed to receive proper feedback from Arduino."
            
        except Exception as e:
            response.success = False
            response.message = f"Error during operation: {e}"
        
        return response

    def wait_for_feedback(self):
        """Wait for feedback from Arduino, blocking until a response is received."""
        start_time = time.time()
        while time.time() - start_time < 1:  # Timeout after 1 seconds
            if self.serial_port.in_waiting > 0:
                feedback = self.serial_port.readline().decode('utf-8').strip()
                if feedback:
                    self.get_logger().info(f"Feedback from Arduino: {feedback}")
                    return feedback
        return None

def main(args=None):
    rclpy.init(args=args)
    util_arduino_serial = ArduinoNode()
    rclpy.spin(util_arduino_serial)
    util_arduino_serial.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
