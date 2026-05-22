import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from chitrak_msgs.msg import ChitrakLegPlanarVelocities
from chitrak_msgs.msg import LegPlanarVelocity
import math


class BodyMotionPlanner(Node):
    def __init__(self):
        super().__init__('body_motion_planner')

        # Geometry parameters
        self.declare_parameter('geometry.length', 0.30)
        self.declare_parameter('geometry.width', 0.20)

        self.L = self.get_parameter('geometry.length').value
        self.W = self.get_parameter('geometry.width').value

        self.cmd_vel = Twist()
        # Set default cmd_vel to avoid errors before the first cmd_vel is received
        self.cmd_vel.linear.x = 0.0
        self.cmd_vel.linear.y = 0.0
        self.cmd_vel.angular.z = 0.0
        self.subscription_ = self.create_subscription(
            Twist,
            '/chitrak/cmd_vel',
            self.cmd_vel_callback,
            10
        )

        self.publisher_ = self.create_publisher(
            ChitrakLegPlanarVelocities,
            '/chitrak/leg_planar_velocities',
            10
        )
        self.timer = self.create_timer(0.1, self.publish_leg_velocities)  # 10 Hz

    def cmd_vel_callback(self, msg):
        self.cmd_vel = msg

    def publish_leg_velocities(self):
        vx = self.cmd_vel.linear.x
        vy = self.cmd_vel.linear.y
        omega = self.cmd_vel.angular.z

        vx = self.cmd_vel.linear.x
        vy = self.cmd_vel.linear.y
        omega = self.cmd_vel.angular.z

        # Inverse differential kinematics
        vr = vx + (self.W / 2.0) * omega
        vl = vx - (self.W / 2.0) * omega
        vf = vy + (self.L / 2.0) * omega
        vb = vy - (self.L / 2.0) * omega

        # Leg speeds and angles
        v_fr = math.hypot(vr, vf)
        v_bl = math.hypot(vl, vb)
        v_fl = math.hypot(vl, vf)
        v_br = math.hypot(vr, vb)

        theta_fr = math.atan2(vf, vr) if v_fr > 0 else 0.0
        theta_bl = math.atan2(vb, vl) if v_bl > 0 else 0.0
        theta_fl = math.atan2(vf, vl) if v_fl > 0 else 0.0
        theta_br = math.atan2(vb, vr) if v_br > 0 else 0.0

        fr_planar_vel = LegPlanarVelocity(magnitude=v_fr, direction=theta_fr)
        bl_planar_vel = LegPlanarVelocity(magnitude=v_bl, direction=theta_bl)
        fl_planar_vel = LegPlanarVelocity(magnitude=v_fl, direction=theta_fl)
        br_planar_vel = LegPlanarVelocity(magnitude=v_br, direction=theta_br)

        msg = ChitrakLegPlanarVelocities()
        msg.front_right = fr_planar_vel
        msg.back_left = bl_planar_vel
        msg.front_left = fl_planar_vel
        msg.back_right = br_planar_vel

        self.publisher_.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    body_motion_planner = BodyMotionPlanner()

    try:
        rclpy.spin(body_motion_planner)
    finally:
        body_motion_planner.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
