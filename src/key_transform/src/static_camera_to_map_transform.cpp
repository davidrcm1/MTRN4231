#include <memory>
#include <string>
#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/static_transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>

// static_camera_to_map_transform


class StaticCameraToMapTransform : public rclcpp::Node
{
public:
    StaticCameraToMapTransform() : Node("static_camera_to_map_transform")
    {
        // Initialize the static transform broadcaster
        static_broadcaster_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(this);
        publishStaticTransform();
    }

private:
    void publishStaticTransform()
    {
        geometry_msgs::msg::TransformStamped transformStamped;

        // Set the header information
        transformStamped.header.stamp = this->get_clock()->now();
        transformStamped.header.frame_id = "base_frame";
        transformStamped.child_frame_id = "camera_frame";

        // Set translation (adjust based on the actual fixed position of the camera relative to the map)
        transformStamped.transform.translation.x = 520.0;  
        transformStamped.transform.translation.y = -990.0;  
        transformStamped.transform.translation.z = 0.0;  

        // Set rotation as an identity quaternion (no rotation)
        transformStamped.transform.rotation.x = 0.0;
        transformStamped.transform.rotation.y = 0.0;
        transformStamped.transform.rotation.z = 0.0;
        transformStamped.transform.rotation.w = 1.0;

        // Broadcast the static transform
        static_broadcaster_->sendTransform(transformStamped);
        RCLCPP_INFO(this->get_logger(), "Published static transform from 'base_frame' to 'camera_frame'");
    }

    std::shared_ptr<tf2_ros::StaticTransformBroadcaster> static_broadcaster_;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<StaticCameraToMapTransform>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
