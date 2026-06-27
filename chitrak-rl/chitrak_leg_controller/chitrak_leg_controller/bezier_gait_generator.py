import rclpy
from rclpy.node import Node
from chitrak_msgs.msg import LegGaitParams
from chitrak_msgs.msg import ChitrakGaitParams
from chitrak_msgs.msg import LegEndPositions
from geometry_msgs.msg import Point
import math
import numpy as np


class BezierGaitGenerator(Node):
    def __init__(self):
        super().__init__('bezier_gait_generator')

        self.gait_params = ChitrakGaitParams()
        # Set default gait params to avoid errors before the first leg gait params are received
        default_leg_gait_params = LegGaitParams(
            step_length=0.0,
            step_direction=0.0,
            step_height=0.0,
            step_frequency=1.0,
            phase_offset=0.0,
            duty_factor=0.8,
            hip_height=0.15
        )
        for leg in ['front_right', 'back_left', 'front_left', 'back_right']:
            setattr(self.gait_params, leg, default_leg_gait_params)
        self.subscription_ = self.create_subscription(
            ChitrakGaitParams,
            '/chitrak/gait_params',
            self.gait_params_callback,
            10
        )
        self.bezier_weights = {}
        self.compute_bezier_weights()

        self.publisher_ = self.create_publisher(LegEndPositions, '/chitrak/leg_end_positions', 10)
        self.timer = self.create_timer(0.01, self.publish_leg_end_positions)  # 100 Hz

    def gait_params_callback(self, msg):
        self.gait_params = msg
        self.compute_bezier_weights()

    def compute_bezier_weights(self):
        for leg in ['front_right', 'back_left', 'front_left', 'back_right']:
            leg_gait_params = getattr(self.gait_params, leg)

            d = leg_gait_params.step_length
            h = leg_gait_params.step_height
            theta = leg_gait_params.step_direction
            hip_height = leg_gait_params.hip_height

            dx = d * 0.7 * np.cos(theta)
            dy = d * 0.7 * np.sin(theta)
            p3x, p3y, p3z = 0, 0, -hip_height

            P = np.array([
                [p3x, p3y, p3z + h],                                     # P0
                [p3x + (4/5) * dx, p3y + (4/5) * dy, p3z + (3/5) * h],   # P1
                [p3x + (5/5) * dx, p3y + (5/5) * dy, p3z + (1/5) * h],   # P2
                [p3x, p3y, p3z],                                         # P3
                [p3x - (5/5) * dx, p3y - (5/5) * dy, p3z + (1/5) * h],   # P4
                [p3x - (4/5) * dx, p3y - (4/5) * dy, p3z + (3/5) * h],   # P5
                [p3x, p3y, p3z + h],                                     # P6
            ])

            U = [0, 0.1, 0.2, 0.5, 0.8, 0.9, 1.0]

            T = np.array([
                [u**0, u**1, u**2, u**3, u**4, u**5, u**6]
                for u in U
            ])

            M = np.array([
                [  1,   0,   0,    0,    0,   0,  0],  # noqa: E201, E203
                [ -6,   6,   0,    0,    0,   0,  0],  # noqa: E201, E203
                [ 15, -30,  15,    0,    0,   0,  0],  # noqa: E201, E203
                [-20,  60, -60,   20,    0,   0,  0],  # noqa: E201, E203
                [ 15, -60,  90,  -60,   15,   0,  0],  # noqa: E201, E203
                [ -6,  30, -60,   60,  -30,   6,  0],  # noqa: E201, E203
                [  1,  -6,  15,  -20,   15,  -6,  1]   # noqa: E201, E203
            ])

            W = np.linalg.inv(M) @ np.linalg.inv(T.T @ T) @ T.T @ P
            self.bezier_weights[leg] = W

    def t_to_u(self, t, leg):
        leg_gait_params = getattr(self.gait_params, leg)
        Tp = 1 / leg_gait_params.step_frequency

        beta_u = 0.6  # Due to the choice of U
        beta_t = leg_gait_params.duty_factor

        norm_t = t / Tp
        if 0 <= norm_t < (1 - beta_t) / 2:
            u = norm_t * (beta_u - 1)/(beta_t - 1)
        elif (1 - beta_t) / 2 <= norm_t < (1 + beta_t) / 2:
            u = (beta_u / (2 * beta_t)) * (2 * norm_t - 1) + 0.5
        else:
            u = (beta_u / 2) - (beta_u - 1) * (1 + beta_t - 2*norm_t) / (2*(beta_t - 1)) + 0.5

        return u

    def compute_bezier_point(self, u, leg):
        W = self.bezier_weights[leg]

        n = 6
        B = np.zeros(3)
        for i in range(n + 1):
            b = math.comb(n, i) * (u ** i) * ((1 - u) ** (n - i))
            B += b * W[i]

        return B

    def publish_leg_end_positions(self):
        msg = LegEndPositions()

        for leg in ['front_right', 'back_left', 'front_left', 'back_right']:
            Tp = 1 / getattr(self.gait_params, leg).step_frequency
            current_time = self.get_clock().now().nanoseconds / 1e9
            phase_offset = getattr(self.gait_params, leg).phase_offset
            t = current_time + phase_offset * Tp
            t %= Tp

            u = self.t_to_u(t, leg)

            end_position = self.compute_bezier_point(u, leg)
            setattr(msg, leg, Point(x=end_position[0], y=end_position[1], z=end_position[2]))

        self.publisher_.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    bezier_gait_generator = BezierGaitGenerator()

    try:
        rclpy.spin(bezier_gait_generator)
    finally:
        bezier_gait_generator.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
