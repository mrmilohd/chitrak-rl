import rclpy
from rclpy.node import Node
from chitrak_msgs.msg import LegAvgVelocity
from chitrak_msgs.msg import LegAvgVelocityArray
from chitrak_msgs.msg import LegGaitParams
from chitrak_msgs.msg import LegGaitParamsArray 

class GaitScheduler(Node):
    def __init__(self):
        super().__init__('gait_scheduler')

        # Gait parameters
        self.declare_parameter('gait.step_height', 2.0)  # TODO: If required, make this dynamic
        self.declare_parameter('gait.min_step_frequency', 0.25)
        self.declare_parameter('gait.max_step_frequency', 1.0)
        self.declare_parameter('gait.max_step_length', 5.0)
        self.declare_parameter('gait.type', 'walk')

        self.step_height = self.get_parameter('gait.step_height').value
        self.min_step_frequency = self.get_parameter('gait.min_step_frequency').value
        self.max_step_frequency = self.get_parameter('gait.max_step_frequency').value
        self.max_step_length = self.get_parameter('gait.max_step_length').value
        self.gait_type = self.get_parameter('gait.type').value

        self.leg_avg_velocities = LegAvgVelocityArray()
        # Set default leg average velocities to avoid errors before the first cmd_vel is received
        self.leg_avg_velocities.leg_avg_velocities = [
            LegAvgVelocity(magnitude=0.0, direction=0.0)
            for _ in range(4)
        ]
        self.subscription_ = self.create_subscription(LegAvgVelocityArray, '/chitrak/leg_avg_velocities', self.leg_avg_velocities_callback, 10)

        self.publisher_ = self.create_publisher(LegGaitParamsArray, '/chitrak/leg_gait_params', 10)
        self.timer = self.create_timer(0.1, self.publish_gait_params) # 10 Hz      

    def leg_avg_velocities_callback(self, msg):
        self.leg_avg_velocities = msg

    def compute_step_params(self, speed):
        if speed < 0.1:
            return 0.0, 0.0

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
        # Leg commands
        v_fr = self.leg_avg_velocities.leg_avg_velocities[0].magnitude
        v_bl = self.leg_avg_velocities.leg_avg_velocities[1].magnitude
        v_fl = self.leg_avg_velocities.leg_avg_velocities[2].magnitude
        v_br = self.leg_avg_velocities.leg_avg_velocities[3].magnitude

        theta_fr = self.leg_avg_velocities.leg_avg_velocities[0].direction
        theta_bl = self.leg_avg_velocities.leg_avg_velocities[1].direction
        theta_fl = self.leg_avg_velocities.leg_avg_velocities[2].direction
        theta_br = self.leg_avg_velocities.leg_avg_velocities[3].direction

        leg_commands = [
            (v_fr, theta_fr),
            (v_bl, theta_bl),
            (v_fl, theta_fl),
            (v_br, theta_br)
        ]

        msg = LegGaitParamsArray()
        msg.leg_gait_params = []

        for i, (v, theta) in enumerate(leg_commands):
            leg_gait_params = LegGaitParams()

            step_length, step_frequency = self.compute_step_params(v)
            leg_gait_params.step_length = step_length
            leg_gait_params.step_frequency = step_frequency
            leg_gait_params.step_direction = theta
            leg_gait_params.step_height = self.step_height

            if self.gait_type == 'walk':
                # FR=0, BL=0.25, FL=0.5, BR=0.75
                leg_gait_params.phase_offset = 0.25 * i
            elif self.gait_type == 'trot':
                # FR and BL in phase, FL and BR in phase
                leg_gait_params.phase_offset = 0.5 * (i % 2)

            # TODO: Make an IMU node calculate this based on teleop commands and robot stability,
            # and subscribe to it here instead of hardcoding it.
            leg_gait_params.hip_height = 15.0

            msg.leg_gait_params.append(leg_gait_params)

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
