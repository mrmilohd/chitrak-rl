import rclpy
from rclpy.node import Node
from chitrak_msgs.msg import LegJointAngles
from chitrak_msgs.msg import ChitrakJointAngles
from std_msgs.msg import Float64


class JointAnglesTranslator(Node):
    def __init__(self):
        super().__init__('joint_angles_translator')

        # Set default angles to avoid errors before the first joint angles are received
        for leg in ['front_right', 'back_left', 'front_left', 'back_right']:
            for joint in ['hip_roll', 'hip_pitch', 'knee']:
                setattr(self, f'{leg}_{joint}_angle', 0.0)
        self.subscription_ = self.create_subscription(
            ChitrakJointAngles,
            '/chitrak/joint_angles',
            self.joint_angles_callback,
            10
        )

        self.publishers_ = {}
        for leg in ['front_right', 'back_left', 'front_left', 'back_right']:
            for joint in ['hip_roll', 'hip_pitch', 'knee']:
                topic_name = f'/chitrak/{leg}_{joint}_angle'
                self.publishers_[f'{leg}_{joint}'] = self.create_publisher(Float64, topic_name, 10)
        self.timer = self.create_timer(0.01, self.publish_joint_angles)  # 100 Hz

    def joint_angles_callback(self, msg):
        for leg in ['front_right', 'back_left', 'front_left', 'back_right']:
            leg_joint_angles = getattr(msg, leg)

            hip_angle = leg_joint_angles.hip_angle
            thigh_angle = leg_joint_angles.thigh_angle
            calf_angle = leg_joint_angles.calf_angle

            hip_roll_angle = hip_angle if 'front' in leg else -hip_angle
            hip_pitch_angle = thigh_angle if 'right' in leg else -thigh_angle
            knee_angle = (
                thigh_angle + calf_angle if 'left' in leg else -(thigh_angle + calf_angle)
            )

            setattr(self, f'{leg}_hip_roll_angle', hip_roll_angle)
            setattr(self, f'{leg}_hip_pitch_angle', hip_pitch_angle)
            setattr(self, f'{leg}_knee_angle', knee_angle)
    
    def publish_joint_angles(self):
        for leg in ['front_right', 'back_left', 'front_left', 'back_right']:
            for joint in ['hip_roll', 'hip_pitch', 'knee']:
                msg = Float64()

                angle = getattr(self, f'{leg}_{joint}_angle')
                msg.data = angle

                self.publishers_[f'{leg}_{joint}'].publish(msg)


def main(args=None):
    rclpy.init(args=args)
    joint_angles_translator = JointAnglesTranslator()
    
    try:
        rclpy.spin(joint_angles_translator)
    finally:
        joint_angles_translator.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
