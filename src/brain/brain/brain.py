import rclpy
from rclpy.node import Node
from key_pos_msgs.msg import KeyPositionArray, KeyPosition, MoveItCoords
from std_srvs.srv import Trigger
import time

# Global z and quaternion values for all key locations
Z_GLOBAL = 20
QX_GLOBAL = 0
QY_GLOBAL = 0
QZ_GLOBAL = 0
QW_GLOBAL = 1

# Define HOME_COORDS as a tuple of x, y, z
HOME_COORDS = (131, -589, -176)

class BrainNode(Node):
    def __init__(self):
        super().__init__('brain_node')

        # Array of key IDs to play
        self.keys_to_play = ["Y1", "Y2"]

        # Store transformed key locations and person detection status
        self.transformed_key_locations = {}
        self.people_detected = False
        self.dots_detected = True

        # Publisher to send MoveitCoords to moveit node
        self.moveit_publisher = self.create_publisher(MoveItCoords, 'moveit_target', 10)

        # Subscribers to key positions and transformed key positions
        self.create_subscription(KeyPositionArray, 'key_positions', self.key_positions_callback, 10)
        self.create_subscription(KeyPositionArray, 'transformed_key_positions', self.transformed_key_positions_callback, 10)

        # Create a client to communicate with the Arduino service
        self.client = self.create_client(Trigger, 'trigger_operation')
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for the Arduino service to become available...")

        # Main loop to process keys
        self.create_timer(1.0, self.robot_loop)

    def key_positions_callback(self, msg):
        """Callback for receiving people_detected status."""
        self.people_detected = msg.people_detected
        self.dots_detected = msg.dots_detected

    def transformed_key_positions_callback(self, msg):
        """Callback for receiving transformed key positions."""
        for key in msg.key_positions:
            self.transformed_key_locations[key.letter] = (key.x, key.y, Z_GLOBAL)

    def robot_loop(self):
        """Main loop to iterate over keys and perform actions."""
        for key_id in self.keys_to_play:
            # Check if people are detected
            if self.people_detected:
                self.get_logger().info("People detected, moving robot to home position...")

                # Create moveit message for robot to go to HOME_COORDS with default orientation
                target_msg = MoveItCoords()
                target_msg.x, target_msg.y, target_msg.z = HOME_COORDS
                target_msg.qx = QX_GLOBAL
                target_msg.qy = QY_GLOBAL
                target_msg.qz = QZ_GLOBAL
                target_msg.qw = QW_GLOBAL

                # Print target_msg for debugging
                self.get_logger().info(f"Target Message for HOME: {target_msg}")

                # Publish the move command to move the robot to home
                self.moveit_publisher.publish(target_msg)
                self.get_logger().info("Published MoveIt target for HOME position")

                # Wait briefly to simulate movement to the home position before continuing
                time.sleep(5)  # Adjust sleep duration as needed
                return  # Skip this loop iteration if people are detected
            elif not self.dots_detected: # TODO get rid of if not working
                self.get_logger().info("Dots not detected, ensure dots are in view for localisation")
                
                return
            else:
                # Get transformed coordinates for the current key_id
                if key_id in self.transformed_key_locations:
                    x, y, z = self.transformed_key_locations[key_id]
                    self.get_logger().info(f"Moving to key {key_id} at position ({x}, {y}, {z})")

                    # Create and publish the MoveitCoords message
                    target_msg = MoveItCoords()
                    target_msg.x = float(x)
                    target_msg.y = float(y)
                    target_msg.z = float(Z_GLOBAL)
                    target_msg.qx = float(QX_GLOBAL)
                    target_msg.qy = float(QY_GLOBAL)
                    target_msg.qz = float(QZ_GLOBAL)
                    target_msg.qw = float(QW_GLOBAL)

                    # Print target_msg for debugging
                    self.get_logger().info(f"Target Message for {key_id}: {target_msg}")

                    self.moveit_publisher.publish(target_msg)
                    self.get_logger().info(f"Published MoveIt target for {key_id}")

                    # Strike the key using the Arduino service
                    self.strike_key(key_id)
                    time.sleep(2)  # Simulate the time taken for the strike
                else:
                    self.get_logger().warning(f"Key '{key_id}' not found in transformed key locations.")

            # Wait a moment to simulate time taken for movement and striking
            time.sleep(10)

    def strike_key(self, key_id):
        """Send a request to the Arduino service to strike the key."""
        self.get_logger().info(f"Striking key {key_id}")
        request = Trigger.Request()
        self.future = self.client.call_async(request)
        self.future.add_done_callback(self.handle_response)

    def handle_response(self, future):
        try:
            response = future.result()
            if response.success:
                self.get_logger().info(f"Arduino operation successful: {response.message}")
            else:
                self.get_logger().error(f"Arduino operation failed: {response.message}")
        except Exception as e:
            self.get_logger().error(f"Service call failed: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    brain_node = BrainNode()
    rclpy.spin(brain_node)
    brain_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
