import rclpy
from rclpy.node import Node
from chitrak_msgs.msg import ChitrakLegPlanarVelocities
from chitrak_msgs.msg import LegPlanarVelocity
from chitrak_msgs.msg import ChitrakGaitParams
from chitrak_msgs.msg import LegGaitParams


class GaitScheduler(Node):
    def __init__(self):
        super().__init__('gait_scheduler')

        # Gait parameters
        self.declare_parameter('gait.step_height', 0.05)  # TODO: If required, make this dynamic
        self.declare_parameter('gait.min_step_frequency', 0.25)
        self.declare_parameter('gait.max_step_frequency', 1.0)
        self.declare_parameter('gait.max_step_length', 0.05)
        self.declare_parameter('gait.duty_factor', 0.8)  # TODO: If required, make this dynamic
        self.declare_parameter('gait.type', 'walk')

        self.step_height = self.get_parameter('gait.step_height').value
        self.min_step_frequency = self.get_parameter('gait.min_step_frequency').value
        self.max_step_frequency = self.get_parameter('gait.max_step_frequency').value
        self.max_step_length = self.get_parameter('gait.max_step_length').value
        self.duty_factor = self.get_parameter('gait.duty_factor').value
        self.gait_type = self.get_parameter('gait.type').value

        self.leg_index = {
            'front_right': 0,
            'back_left': 1,
            'front_left': 2,
            'back_right': 3
        }

        self.leg_planar_velocities = ChitrakLegPlanarVelocities()
        # Set default leg average velocities to avoid errors before the first cmd_vel is received
        for leg in ['front_right', 'back_left', 'front_left', 'back_right']:
            setattr(
                self.leg_planar_velocities, leg, LegPlanarVelocity(magnitude=0.0, direction=0.0)
            )
        self.subscription_ = self.create_subscription(
            ChitrakLegPlanarVelocities,
            '/chitrak/leg_planar_velocities',
            self.leg_planar_velocities_callback,
            10
        )

        self.publisher_ = self.create_publisher(ChitrakGaitParams, '/chitrak/gait_params', 10)
        self.timer = self.create_timer(0.1, self.publish_gait_params)  # 10 Hz

    def leg_planar_velocities_callback(self, msg):
        self.leg_planar_velocities = msg

    def compute_step_params(self, speed):
        if speed < 0.01:
            return 0.0, self.min_step_frequency

        # try max frequency first
        f = self.max_step_frequency
        d = speed / f

        if d > self.max_step_length:
            d = self.max_step_length
            f = speed / d

            # clamp frequency also
            f = min(max(f, self.min_step_frequency), self.max_step_frequency)

        return d, f

    def publish_gait_params(self):
        msg = ChitrakGaitParams()

        for leg in ['front_right', 'back_left', 'front_left', 'back_right']:
            v = getattr(self.leg_planar_velocities, leg).magnitude
            theta = getattr(self.leg_planar_velocities, leg).direction

            leg_gait_params = LegGaitParams()

            step_length, step_frequency = self.compute_step_params(v)
            leg_gait_params.step_length = step_length
            leg_gait_params.step_direction = theta
            leg_gait_params.step_height = self.step_height

            leg_gait_params.step_frequency = step_frequency
            if self.gait_type == 'walk':
                # FR=0, BL=0.25, FL=0.5, BR=0.75
                leg_gait_params.phase_offset = 0.25 * self.leg_index[leg]
            elif self.gait_type == 'trot':
                # FR and BL in phase, FL and BR in phase
                leg_gait_params.phase_offset = 0.5 * (self.leg_index[leg] % 2)
            leg_gait_params.duty_factor = self.duty_factor

            # TODO: Make an IMU node calculate this based on teleop commands and robot stability,
            # and subscribe to it here instead of hardcoding it.
            leg_gait_params.hip_height = 0.15

            setattr(msg, leg, leg_gait_params)

        self.publisher_.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    gait_scheduler = GaitScheduler()

    try:
        rclpy.spin(gait_scheduler)
    finally:
        gait_scheduler.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
