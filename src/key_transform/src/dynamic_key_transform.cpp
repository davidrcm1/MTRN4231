#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <key_pos_msgs/msg/key_position_array.hpp>

// dynamic_key_transform


class DynamicKeyTransform : public rclcpp::Node
{
public:
    DynamicKeyTransform() : Node("dynamic_key_transform")
    {
        // Initialize Transform Broadcaster
        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

        // Subscribe to key positions from key_pub
        subscription_ = this->create_subscription<key_pos_msgs::msg::KeyPositionArray>(
            "key_positions",
            10,
            std::bind(&DynamicKeyTransform::keyPositionCallback, this, std::placeholders::_1));
    }

private:
    void keyPositionCallback(const key_pos_msgs::msg::KeyPositionArray::SharedPtr msg)
    {
        // For each key, create and publish a transform relative to camera_frame
        for (const auto &key_pos : msg->key_positions)
        {
            publishTransform(
                key_pos.x, key_pos.y, 0.0,  // Assuming keys are on a plane at z = 0
                key_pos.letter,
                "camera_frame");
        }
    }

    void publishTransform(double x, double y, double z, const std::string &key_id, const std::string &parent_frame)
    {
        geometry_msgs::msg::TransformStamped transformStamped;

        // Set up the header informations
        transformStamped.header.stamp = this->get_clock()->now();
        transformStamped.header.frame_id = parent_frame;
        transformStamped.child_frame_id = "key_" + key_id;

        // Set the translation and identity rotation
        transformStamped.transform.translation.x = x;
        transformStamped.transform.translation.y = y;
        transformStamped.transform.translation.z = z;
        transformStamped.transform.rotation.x = 0.0;
        transformStamped.transform.rotation.y = 0.0;
        transformStamped.transform.rotation.z = 0.0;
        transformStamped.transform.rotation.w = 1.0;

        // Publish the transform
        tf_broadcaster_->sendTransform(transformStamped);
        RCLCPP_INFO(this->get_logger(), "Published transform for key %s at (%f, %f, %f) in %s frame", key_id.c_str(), x, y, z, parent_frame.c_str());
    }

    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    rclcpp::Subscription<key_pos_msgs::msg::KeyPositionArray>::SharedPtr subscription_;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<DynamicKeyTransform>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
