import numpy as np
from ROAR.planning_module.local_planner.loop_simple_waypoint_following_local_planner import LoopSimpleWaypointFollowingLocalPlanner
from ROAR.planning_module.local_planner.local_planner import LocalPlanner
from ROAR.utilities_module.data_structures_models import Transform, Rotation, Location
from ROAR.utilities_module.vehicle_models import Vehicle, VehicleControl
from ROAR.control_module.controller import Controller
from ROAR.planning_module.mission_planner.mission_planner import MissionPlanner
from ROAR.planning_module.behavior_planner.behavior_planner import BehaviorPlanner

import logging
from typing import Union
from ROAR.utilities_module.errors import (
    AgentException,
)
from ROAR.agent_module.agent import Agent
# from ROAR.agent_module.mark_agent import MarkAgent
import json
from pathlib import Path
from collections import deque

class DynamicWindowsApproach(LoopSimpleWaypointFollowingLocalPlanner):
    def __init__(
            self,
            agent: Agent,
            controller: Controller
    ):
        self.agent = agent
        self.controller = controller
        self.old_waypoint = None
        """
        Initialize Dynamics Windows Approach Planner
        Args:
            agent: newest agent state
            controller: Control module used
        """
        super().__init__(agent=self.agent, controller=self.controller, mission_planner=self.agent.mission_planner,
                         behavior_planner=self.agent.behavior_planner, closeness_threshold=1)
        self.logger = logging.getLogger("DynamicsWindowsApproach")
        self.logger.debug("Dynamics Windows Approach Initiated")

    def run_in_series(self) -> VehicleControl:
        simplewaypointcontrol, targetwaypoint = super().run_in_series_for_dwa() # Gives control/target waypoint that the thinks should do next from simple_waypoint_following_local_planner
        # if self.old_waypoint is None:
        #     self.old_waypoint = [targetwaypoint.location.x, targetwaypoint.location.z]
        vehicle_transform = self.agent.vehicle.transform
        vehicle_velo = self.agent.vehicle.velocity
        vehicle_control = self.agent.vehicle.control  # this is the PREVIOUS control
        max_speed = self.agent.agent_settings.max_speed # 20.0
        # dynamic_window = self.calc_dynamic_window(vehicle_velo, vehicle_control, max_speed)
        dynamic_window = 0
        # # print("dynamicwindow", dynamic_window)
        next_waypoint = self.calc_control_and_trajectory(dynamic_window, vehicle_velo, vehicle_control, vehicle_transform, targetwaypoint, max_speed)
        # #

        formatted_nw = Transform(location = Location(x = next_waypoint[0], y = 0, z = next_waypoint[1]), rotation = targetwaypoint.rotation)
        #
        control: VehicleControl = self.controller.run_in_series(next_waypoint=formatted_nw)
        # # print("\n")
        # # print("formatted_nw", formatted_nw, "\n")
        print("targetwaypoint", simplewaypointcontrol, targetwaypoint, "\n")
        # # print("\n")
        return control
        # return simplewaypointcontrol
    def calc_dynamic_window(self, vehicle_velo, vehicle_control, max_speed):
        # Vs = [config.min_speed, config.max_speed, -config.max_yaw_rate, config.max_yaw_rate]
        # x = initial state [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
        dt = 0.25
        acceleration_coefficient = 5
        steering_coefficent = 0.01
        yvelo = -1 * vehicle_velo.y #negative when going forward so make it positive
        steering = vehicle_control.steering
        print("steering", steering)
        Vs = [0.0, max_speed/2, -1, 1]  # max speed is 20.0
        Vs = [0.1, .2, -1.0, 1.0]
        Vd = [yvelo -acceleration_coefficient * dt, yvelo +acceleration_coefficient * dt, steering -steering_coefficent * dt, steering +steering_coefficent * dt]
        if Vd[2] > 1:
            Vd[2] = 1
        if Vd[2] < -1:
            Vd[2] = -1
        if Vd[3] > 1:
            Vd[3] = 1
        if Vd[3] < -1:
            Vd[3] = -1
        dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]), max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]
        # print(dw)
        dw = Vs
        return dw

    def calc_control_and_trajectory(self, dw, vehicle_velo, vehicle_control, vehicle_transform, goal, max_speed):
        min_cost = np.inf
        waypoint = goal
        steer = vehicle_control.steering
        yvelo = vehicle_velo.y
        x, z, yaw = vehicle_transform.location.x,vehicle_transform.location.z, vehicle_transform.rotation.yaw
        initial_state = [x, z, np.deg2rad(yaw), yvelo, steer]  #yaw given in degrees so change to radians, negative y is forward direction
        obstacles = self.agent.occupancy_map.get_map(goal, view_size=(100, 100),  vehicle_value = -10)
        # x = initial state [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
        for v in np.arange(0.01, 7,1): # v is velocity
            for y in np.arange(-180, 180, 30):  # y is steering rate
                estimated_goal = self.predict_trajectory(initial_state, v, np.deg2rad(y))

                to_goal_cost_gain = 1.0 #0.15
                # speed_cost_gain = 0.1
                obstacle_cost_gain = 0.0
                # smooth_cost_gain = 0.50
        #
                to_goal_cost = to_goal_cost_gain * self.calc_to_goal_cost(goal, estimated_goal)
                # speed_cost = speed_cost_gain * (max_speed - estimated_goal[3])
                ob_cost = obstacle_cost_gain * self.calc_obstacle_cost(estimated_goal, obstacles)    #how to get obstacles
                # smooth_cost = smooth_cost_gain * self.calc_smooth_cost(estimated_goal)
                final_cost = to_goal_cost
                # print("estimatedgoal", estimated_goal, final_cost)

        #         print("final cost", final_cost)
        #         # search for minimum cost
                if min_cost > final_cost:
                    min_cost = final_cost
                    waypoint = estimated_goal
                # print("newmin", min_cost)
        print("dw", dw)
        print("chosen estimated goal", waypoint)
        # print(self.agent.occupancy_map.get_map(Transform(location=Location(x=estimated_goal[0], y=0, z=estimated_goal[1])), view_size=(8, 4)))
        print("actual", goal)
        print("min_cost", min_cost)
        print("called")
        # self.old_waypoint = waypoint[0:2]
        return waypoint

    def calc_smooth_cost(self, estimated_goal):
        """
        calc obstacle cost inf: collision
        """
        # print("estimatedgoal", estimated_goal)
        # print("old", self.old_waypoint)
        cost = np.linalg.norm(self.old_waypoint - estimated_goal[0:2])
        # print(cost)
        return cost

    def calc_obstacle_cost(self, estimated_goal, obstacles):
        """
        calc obstacle cost inf: collision
        """
        x = obstacles[0:2, :]
        return 0


    def calc_to_goal_cost(self, goal, estimated_goal):
        """
            calc to goal cost with angle difference
        """
        dx = abs(goal.location.x - estimated_goal[0])
        # dx = dx*2
        dy = abs(goal.location.z - estimated_goal[1])
        cost = np.sqrt((dx*dx) + (dy*dy))
        return cost

    # def motion(self, x, u, dt):
    #     """
    #     motion model
    #     # x = initial state [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
    #     """
    #
    #     x[2] = u[1]  # need to update yaw angle first
    #     x[1] += -u[0] * np.cos(x[2])*6.944444e-5
    #     x[0] += u[0] * np.sin(x[2]) *6.944444e-5
    #     x[3] = u[0]
    #     x[4] = u[1]/dt
    #
    #     return x

    def predict_trajectory(self, initial_state, v, y):
        """
        predict trajectory with an input
        """

        x = np.array(initial_state)
        x[2] = y  # need to update yaw angle first
        x[1] += -v * np.cos(x[2])
        x[0] += v * np.sin(x[2])
        x[3] = v
        x[4] = y / 0.25
        return x


