// Forward key presses made in the Gazebo window to ROS 2.
//
// Gazebo's KeyboardGUIPlugin publishes every key pressed while the 3D view has
// focus on the gz-transport topic ~/keyboard/keypress (the value is the Qt key
// code).  This world plugin republishes those codes on /wiloc/gazebo_keys
// (std_msgs/Int32), where the WiLoc dashboard turns them into robot commands,
// so the robot can be driven from the Gazebo window as well as the dashboard.

#include <memory>

#include <gazebo/common/Plugin.hh>
#include <gazebo/msgs/msgs.hh>
#include <gazebo/physics/World.hh>
#include <gazebo/transport/transport.hh>
#include <gazebo_ros/node.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/int32.hpp>

namespace wiloc_gazebo
{

class KeyForwarder : public gazebo::WorldPlugin
{
public:
  void Load(gazebo::physics::WorldPtr world, sdf::ElementPtr sdf) override
  {
    ros_node_ = gazebo_ros::Node::Get(sdf);
    pub_ = ros_node_->create_publisher<std_msgs::msg::Int32>("/wiloc/gazebo_keys", 20);

    gz_node_ = gazebo::transport::NodePtr(new gazebo::transport::Node());
    gz_node_->Init(world->Name());
    sub_ = gz_node_->Subscribe("~/keyboard/keypress", &KeyForwarder::OnKey, this);
    RCLCPP_INFO(ros_node_->get_logger(), "forwarding Gazebo key presses to /wiloc/gazebo_keys");
  }

private:
  void OnKey(ConstAnyPtr & msg)
  {
    std_msgs::msg::Int32 out;
    out.data = msg->int_value();
    pub_->publish(out);
  }

  gazebo_ros::Node::SharedPtr ros_node_;
  rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr pub_;
  gazebo::transport::NodePtr gz_node_;
  gazebo::transport::SubscriberPtr sub_;
};

GZ_REGISTER_WORLD_PLUGIN(KeyForwarder)

}  // namespace wiloc_gazebo
