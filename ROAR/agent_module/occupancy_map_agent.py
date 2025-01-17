from ROAR.agent_module.agent import Agent
from ROAR.utilities_module.data_structures_models import SensorsData
from ROAR.utilities_module.vehicle_models import Vehicle, VehicleControl
from ROAR.configurations.configuration import Configuration as AgentConfig
from ROAR.perception_module.ground_plane_detector import GroundPlaneDetector
from ROAR.perception_module.depth_to_pointcloud_detector import DepthToPointCloudDetector
import numpy as np
from ROAR.control_module.pid_controller import PIDController
from ROAR.planning_module.local_planner.simple_waypoint_following_local_planner import \
    SimpleWaypointFollowingLocalPlanner
from ROAR.planning_module.behavior_planner.behavior_planner import BehaviorPlanner
from ROAR.planning_module.mission_planner.waypoint_following_mission_planner import WaypointFollowingMissionPlanner
from ROAR.perception_module.obstacle_detector import ObstacleDetector
from pathlib import Path
from ROAR.utilities_module.occupancy_map import OccupancyGridMap
import matplotlib.pyplot as plt
import open3d as o3d
import cv2
from ROAR.perception_module.legacy.point_cloud_detector import PointCloudDetector
from ROAR.perception_module.obstacle_from_depth import ObstacleFromDepth
import time


class OccupancyMapAgent(Agent):
    def __init__(self, vehicle: Vehicle, agent_settings: AgentConfig, **kwargs):
        super().__init__(vehicle, agent_settings, **kwargs)
        self.route_file_path = Path(self.agent_settings.waypoint_file_path)
        self.pid_controller = PIDController(agent=self, steering_boundary=(-1, 1), throttle_boundary=(0, 1))
        self.mission_planner = WaypointFollowingMissionPlanner(agent=self)
        # initiated right after mission plan
        self.behavior_planner = BehaviorPlanner(agent=self)
        self.local_planner = SimpleWaypointFollowingLocalPlanner(
            agent=self,
            controller=self.pid_controller,
            mission_planner=self.mission_planner,
            behavior_planner=self.behavior_planner,
            closeness_threshold=1.5
        )
        self.occupancy_map = OccupancyGridMap(absolute_maximum_map_size=800,
                                              world_coord_resolution=1,
                                              occu_prob=0.99,
                                              max_points_to_convert=10000,
                                              threaded=True)
        self.obstacle_from_depth_detector = ObstacleFromDepth(agent=self,
                                                              threaded=True,
                                                              max_detectable_distance=0.5,
                                                              max_points_to_convert=20000,
                                                              min_obstacle_height=2)
        self.add_threaded_module(self.obstacle_from_depth_detector)
        self.add_threaded_module(self.occupancy_map)
        # self.vis = o3d.visualization.Visualizer()
        # self.vis.create_window(width=500, height=500)
        # self.pcd = o3d.geometry.PointCloud()
        # self.points_added = False

    def run_step(self, sensors_data: SensorsData, vehicle: Vehicle) -> VehicleControl:
        super().run_step(sensors_data=sensors_data, vehicle=vehicle)
        control = self.local_planner.run_in_series()
        option = "obstacle_coords"  # ground_coords, point_cloud_obstacle_from_depth
        if self.kwargs.get(option, None) is not None:
            points = self.kwargs[option]
            self.occupancy_map.update_async(points)
            arb_points = [self.local_planner.way_points_queue[0].location]
            m = self.occupancy_map.get_map(transform=self.vehicle.transform,
                                           view_size=(200, 200),
                                           vehicle_value=-1,
                                           arbitrary_locations=arb_points,
                                           arbitrary_point_value=-5)
            # print(np.where(m == -5))
            # cv2.imshow("m", m)
            # cv2.waitKey(1)
            # occu_map_vehicle_center = np.array(list(zip(*np.where(m == np.min(m))))[0])
            # correct_next_waypoint_world = self.local_planner.way_points_queue[0]
            # diff = np.array([correct_next_waypoint_world.location.x,
            #                  correct_next_waypoint_world.location.z]) - \
            #        np.array([self.vehicle.transform.location.x,
            #                  self.vehicle.transform.location.z])
            # correct_next_waypoint_occu = occu_map_vehicle_center + diff
            # correct_next_waypoint_occu = np.array([49.97, 44.72596359])
            # estimated_world_coord = self.occupancy_map.cropped_occu_to_world(
            #     cropped_occu_coord=correct_next_waypoint_occu, vehicle_transform=self.vehicle.transform,
            #     occu_vehicle_center=occu_map_vehicle_center)
            # print(f"correct o-> {correct_next_waypoint_occu}"
            #       f"correct w-> {correct_next_waypoint_world.location} | "
            #       f"estimated = {estimated_world_coord.location.x}")
            # cv2.imshow("m", m)
            # cv2.waitKey(1)
            # print()

            # if self.points_added is False:
            #     self.pcd = o3d.geometry.PointCloud()
            #     point_means = np.mean(points, axis=0)
            #     self.pcd.points = o3d.utility.Vector3dVector(points - point_means)
            #     self.vis.add_geometry(self.pcd)
            #     self.vis.poll_events()
            #     self.vis.update_renderer()
            #     self.points_added = True
            # else:
            #     point_means = np.mean(points, axis=0)
            #     self.pcd.points = o3d.utility.Vector3dVector(points - point_means)
            #     self.vis.update_geometry(self.pcd)
            #     self.vis.poll_events()
            #     self.vis.update_renderer()

        if self.local_planner.is_done():
            self.mission_planner.restart()
            self.local_planner.restart()

        return control
