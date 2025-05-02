#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <key_pos_msgs/msg/key_position_array.hpp>
#include <key_pos_msgs/msg/key_position.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

class KeyTransformListener : public rclcpp::Node
{
public:
    KeyTransformListener() : Node("key_transform_listener")
    {
        // Initialize tf2 buffer and listener
        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        // Publisher for transformed key positions in the base frame
        publisher_ = this->create_publisher<key_pos_msgs::msg::KeyPositionArray>("transformed_key_positions", 10);

        // Subscribe to key positions relative to the camera
        subscription_ = this->create_subscription<key_pos_msgs::msg::KeyPositionArray>(
            "key_positions",
            10,
            std::bind(&KeyTransformListener::keyPositionCallback, this, std::placeholders::_1));
    }

private:
    void keyPositionCallback(const key_pos_msgs::msg::KeyPositionArray::SharedPtr msg)
    {
        key_pos_msgs::msg::KeyPositionArray transformed_positions;
        transformed_positions.people_detected = msg->people_detected;

        for (const auto &key_pos : msg->key_positions)
        {
            try
            {
                // Define a PointStamped in the camera frame
                geometry_msgs::msg::PointStamped point_in_camera;
                point_in_camera.header.frame_id = "camera_frame";
                point_in_camera.header.stamp = this->get_clock()->now();
                point_in_camera.point.x = key_pos.x;
                point_in_camera.point.y = key_pos.y;
                point_in_camera.point.z = 0.0;  // Assuming z = 0 for camera plane

                // Transform to the base frame
                geometry_msgs::msg::PointStamped point_in_base = tf_buffer_->transform(point_in_camera, "base_frame");

                // Store the transformed position
                key_pos_msgs::msg::KeyPosition transformed_key_pos;
                transformed_key_pos.letter = key_pos.letter;
                transformed_key_pos.x = point_in_base.point.x;
                transformed_key_pos.y = point_in_base.point.y;
                transformed_positions.key_positions.push_back(transformed_key_pos);
            }
            catch (const tf2::TransformException &ex)
            {
                RCLCPP_WARN(this->get_logger(), "Could not transform key %s: %s", key_pos.letter.c_str(), ex.what());
            }
        }

        // Publish the transformed key positions in base frame
        publisher_->publish(transformed_positions);
        RCLCPP_INFO(this->get_logger(), "Published transformed key positions in base frame");
    }

    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    rclcpp::Publisher<key_pos_msgs::msg::KeyPositionArray>::SharedPtr publisher_;
    rclcpp::Subscription<key_pos_msgs::msg::KeyPositionArray>::SharedPtr subscription_;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<KeyTransformListener>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
