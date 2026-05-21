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

        self.subscription_ = self.create_subscription(LegEndPositions, '/chitrak/leg_end_positions', self.leg_end_positions_callback, 10)
        self.publisher_ = self.create_publisher(MarkerArray, '/chitrak/leg_trajectory_markers', 10)

        self.marker_ids = {
            'front_right': 0,
            'back_left': 1,
            'front_left': 2,
            'back_right': 3,
        }

        self.frame_map = {
            'front_right': 'fr_thigh_link',
            'back_left': 'bl_thigh_link',
            'front_left': 'fl_thigh_link',
            'back_right': 'br_thigh_link',
        }

        self.colors = {
            'front_right': (0.8, 0.8, 0.8, 0.8),
            'back_left':   (0.8, 0.8, 0.8, 0.8),
            'front_left':  (0.8, 0.8, 0.8, 0.8),
            'back_right':  (0.8, 0.8, 0.8, 0.8),
        }

        self.point_history = {
            'front_right': deque(maxlen=100),
            'back_left': deque(maxlen=100),
            'front_left': deque(maxlen=100),
            'back_right': deque(maxlen=100),
        }

    def create_marker(self, leg_name):
        marker = Marker()
        
        marker.header.frame_id = self.frame_map[leg_name]
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

            self.point_history[leg].append(Point(x=point.x, y=point.y, z=point.z))

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
