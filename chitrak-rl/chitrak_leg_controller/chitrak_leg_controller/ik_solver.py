import rclpy
from rclpy.node import Node
from chitrak_msgs.msg import LegEndPositions
from chitrak_msgs.msg import LegJointAngles
from chitrak_msgs.msg import ChitrakJointAngles
from geometry_msgs.msg import Point
import math


class IKSolver(Node):
    def __init__(self):
        super().__init__('ik_solver')

        # Geometry parameters
        self.declare_parameter('geometry.thigh_length', 0.13)
        self.declare_parameter('geometry.calf_length', 0.15)

        self.thigh_length = self.get_parameter('geometry.thigh_length').value
        self.calf_length = self.get_parameter('geometry.calf_length').value

        self.leg_end_positions = LegEndPositions()
        # Set default leg end positions to avoid errors before the first leg end positions are received
        default_leg_end_position = Point(x=0.0, y=0.0, z=-0.15)
        for leg in ['front_right', 'back_left', 'front_left', 'back_right']:
            setattr(self.leg_end_positions, leg, default_leg_end_position)
        self.subscription_ = self.create_subscription(
            LegEndPositions,
            '/chitrak/leg_end_positions',
            self.leg_end_positions_callback,
            10
        )

        self.publisher_ = self.create_publisher(ChitrakJointAngles, '/chitrak/joint_angles', 10)
        self.timer = self.create_timer(0.01, self.publish_joint_angles)  # 100 Hz

    def leg_end_positions_callback(self, msg):
        self.leg_end_positions = msg

    def solve_ik(self, leg):
        """
        Please don't make any LLM/Coding agent change this function. If you want to change the IK logic, solve it
        BY HAND first!
        """
        end_pos = getattr(self.leg_end_positions, leg)
        x, y, z = end_pos.x, end_pos.y, end_pos.z

        # TODO: check if the position is reachable and handle unreachable cases
        if z == 0:
            if y == 0:
                hip_angle = 0.0
            else:
                hip_angle = math.pi / 2.0 if y > 0 else -math.pi / 2.0
        else:
            hip_angle = math.atan2(y, -z)

        # Planar reduction
        z_plane = -math.sqrt(y**2 + z**2)
        x_plane = x

        A = x_plane
        B = z_plane
        R = math.sqrt(A**2 + B**2)

        phi  = math.atan2(-B, A)
        alpha = math.acos((R**2 + self.calf_length**2 - self.thigh_length**2) / (2 * R * self.calf_length))

        calf_angle = phi - alpha  # Elbow configuration

        x_t = x_plane - self.calf_length * math.cos(calf_angle)
        z_t = z_plane + self.calf_length * math.sin(calf_angle)

        thigh_angle = math.atan2(-z_t, -x_t)

        return hip_angle, thigh_angle, calf_angle
    
    def publish_joint_angles(self):
        msg = ChitrakJointAngles()

        for leg in ['front_right', 'back_left', 'front_left', 'back_right']:
            hip_angle, thigh_angle, calf_angle = self.solve_ik(leg)

            leg_joint_angles = LegJointAngles()
            leg_joint_angles.hip_angle = hip_angle
            leg_joint_angles.thigh_angle = thigh_angle
            leg_joint_angles.calf_angle = calf_angle

            setattr(msg, leg, leg_joint_angles)

        self.publisher_.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    ik_solver = IKSolver()
    
    try:
        rclpy.spin(ik_solver)
    finally:
        ik_solver.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
