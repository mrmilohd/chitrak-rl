import rclpy
from rclpy.node import Node
from chitrak_msgs.msg import LegEndPositions
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Point
from collections import deque


class VizLegTrajectories(Node):
    def __init__(self):
        super().__init__('viz_leg_trajectories')

        self.subscription_ = self.create_subscription(
            LegEndPositions,
            '/chitrak/leg_end_positions',
            self.leg_end_positions_callback,
            10
        )
        self.publisher_ = self.create_publisher(MarkerArray, '/chitrak/leg_trajectory_markers', 10)

        self.marker_ids = {
            'front_right': 0,
            'back_left': 1,
            'front_left': 2,
            'back_right': 3,
        }

        self.offset_map = {
            'front_right': (0.125, -0.076, -0.018),
            'back_left': (-0.145, 0.076, -0.018),
            'front_left': (0.125, 0.076, -0.018),
            'back_right': (-0.145, -0.076, -0.018),
        }

        self.colors = {
            'front_right': (1.0, 0.5, 0.0, 0.8),
            'back_left':   (1.0, 0.5, 0.0, 0.8),
            'front_left':  (0.2, 0.5, 1.0, 0.8),
            'back_right':  (0.2, 0.5, 1.0, 0.8),
        }

        self.point_history = {
            'front_right': deque(maxlen=100),
            'back_left': deque(maxlen=100),
            'front_left': deque(maxlen=100),
            'back_right': deque(maxlen=100),
        }

    def create_marker(self, leg_name):
        marker = Marker()

        marker.header.frame_id = 'base_link'
        marker.header.stamp.sec = 0
        marker.header.stamp.nanosec = 0
        marker.ns = 'leg_trajectories'
        marker.id = self.marker_ids[leg_name]
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        marker.points = list(self.point_history[leg_name])
        marker.scale.x = 0.005  # Line width

        r, g, b, a = self.colors[leg_name]
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.color.a = a

        return marker

    def leg_end_positions_callback(self, msg):
        marker_array = MarkerArray()

        for leg in ['front_right', 'back_left', 'front_left', 'back_right']:
            point = getattr(msg, leg)
            self.point_history[leg].append(
                Point(
                    x=point.x + self.offset_map[leg][0],
                    y=point.y + self.offset_map[leg][1],
                    z=point.z + self.offset_map[leg][2],
                )
            )

        for leg in ['front_right', 'back_left', 'front_left', 'back_right']:
            marker = self.create_marker(leg)
            marker_array.markers.append(marker)

        self.publisher_.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    viz_leg_trajectories = VizLegTrajectories()

    try:
        rclpy.spin(viz_leg_trajectories)
    finally:
        viz_leg_trajectories.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
