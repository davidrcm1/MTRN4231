#include <memory>
#include <rclcpp/rclcpp.hpp>
#include "key_pos_msgs/msg/move_it_coords.hpp" // Ensure correct header name and path
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <geometry_msgs/msg/pose.hpp>

using namespace std::chrono_literals;

class MoveToMarker : public rclcpp::Node {
public:
    MoveToMarker() : Node("move_to_marker") {
        // Subscription to receive coordinates
        target_subscription_ = this->create_subscription<key_pos_msgs::msg::MoveItCoords>(
            "moveit_coords", 10, std::bind(&MoveToMarker::moveToTargetCallback, this, std::placeholders::_1));

        // Initialize MoveGroupInterface with planning settings
        move_group_interface_ = std::make_unique<moveit::planning_interface::MoveGroupInterface>(
            std::shared_ptr<rclcpp::Node>(this), "ur_manipulator"
        );
        move_group_interface_->setPlanningTime(10.0);
        move_group_interface_->setMaxVelocityScalingFactor(0.3);
        move_group_interface_->setMaxAccelerationScalingFactor(0.3);
        move_group_interface_->setNumPlanningAttempts(10);

        RCLCPP_INFO(this->get_logger(), "x");
    }

private:
    void moveToTargetCallback(const key_pos_msgs::msg::MoveItCoords::SharedPtr msg) {
        geometry_msgs::msg::Pose target_pose;
        target_pose.position.x = msg->x;
        target_pose.position.y = msg->y;
        target_pose.position.z = msg->z;
        target_pose.orientation.x = msg->qx;
        target_pose.orientation.y = msg->qy;
        target_pose.orientation.z = msg->qz;
        target_pose.orientation.w = msg->qw;

        RCLCPP_INFO(this->get_logger(), "Move data received: x=%f, y=%f, z=%f", msg->x, msg->y, msg->z);

        move_group_interface_->setPoseTarget(target_pose);
        moveit::planning_interface::MoveGroupInterface::Plan plan;
        if (move_group_interface_->plan(plan) == moveit::core::MoveItErrorCode::SUCCESS) {
            move_group_interface_->execute(plan);
            RCLCPP_INFO(this->get_logger(), "Move successful");
        } else {
            RCLCPP_WARN(this->get_logger(), "Move failed");
        }
    }

    rclcpp::Subscription<key_pos_msgs::msg::MoveItCoords>::SharedPtr target_subscription_;
    std::unique_ptr<moveit::planning_interface::MoveGroupInterface> move_group_interface_;
};

int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<MoveToMarker>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
