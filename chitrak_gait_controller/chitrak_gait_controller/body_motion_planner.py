import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from chitrak_msgs.msg import LegAvgVelocity
from chitrak_msgs.msg import LegAvgVelocityArray
import math

class BodyMotionPlanner(Node):
    def __init__(self):
        super().__init__('body_motion_planner')
        
        # Geometry parameters
        self.declare_parameter('geometry.length', 30.0)
        self.declare_parameter('geometry.width', 20.0)

        self.L = self.get_parameter('geometry.length').value
        self.W = self.get_parameter('geometry.width').value

        self.last_cmd_vel = Twist()
        # Set default cmd_vel to avoid errors before the first cmd_vel is received
        self.last_cmd_vel.linear.x = 0.0
        self.last_cmd_vel.linear.y = 0.0
        self.last_cmd_vel.angular.z = 0.0
        self.subscription_ = self.create_subscription(Twist, '/chitrak/cmd_vel', self.cmd_vel_callback, 10)

        self.publisher_ = self.create_publisher(LegAvgVelocityArray, '/chitrak/leg_avg_velocities', 10)
        self.timer = self.create_timer(0.1, self.publish_leg_velocities) # 10 Hz

    def cmd_vel_callback(self, msg):
        self.last_cmd_vel = msg

    def publish_leg_velocities(self):
        vx = self.last_cmd_vel.linear.x
        vy = self.last_cmd_vel.linear.y
        omega = self.last_cmd_vel.angular.z

        vx = self.last_cmd_vel.linear.x
        vy = self.last_cmd_vel.linear.y
        omega = self.last_cmd_vel.angular.z

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

        theta_fr = math.degrees(math.atan2(vf, vr)) if v_fr > 0 else 0.0
        theta_bl = math.degrees(math.atan2(vb, vl)) if v_bl > 0 else 0.0
        theta_fl = math.degrees(math.atan2(vf, vl)) if v_fl > 0 else 0.0
        theta_br = math.degrees(math.atan2(vb, vr)) if v_br > 0 else 0.0

        fr_avg_vel = LegAvgVelocity(magnitude=v_fr, direction=theta_fr)
        bl_avg_vel = LegAvgVelocity(magnitude=v_bl, direction=theta_bl)
        fl_avg_vel = LegAvgVelocity(magnitude=v_fl, direction=theta_fl)
        br_avg_vel = LegAvgVelocity(magnitude=v_br, direction=theta_br)

        msg = LegAvgVelocityArray()
        msg.leg_avg_velocities = [fr_avg_vel, bl_avg_vel, fl_avg_vel, br_avg_vel]

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
