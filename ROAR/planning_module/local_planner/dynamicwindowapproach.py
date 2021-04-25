import numpy as np

from ROAR.planning_module.local_planner.local_planner import LocalPlanner
from ROAR.utilities_module.data_structures_models import Transform
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
import json
from pathlib import Path
from collections import deque

class DynamicsWindowsApproach(LocalPlanner):
    def __init__(
            self,
            agent: Agent,
            controller: Controller,
    ):
        """
        Initialize Simple Waypoint Following Planner
        Args:
            agent: newest agent state
            controller: Control module used
        """
        super().__init__(agent=agent, controller=controller)
        self.logger = logging.getLogger("DynamicsWindowsApproach")
        self.logger.debug("Dynamics Windows Approach Initiated")
        self.way_points_queue = deque(maxlen=10)

    def is_done(self) -> bool:
        """
        If there are nothing in self.way_points_queue,
        that means you have finished a lap, you are done

        Returns:
            True if Done, False otherwise
        """
        return False

    def run_in_series(self) -> VehicleControl:

        vehicle_transform = self.agent.vehicle.transform.to_array()[1]
        print(vehicle_transform)
        vehicle_velo = self.agent.vehicle.transform.to_array()[0]
        vehicle_control = self.agent.vehicle.transform.to_array()[2]
        max_speed = self.agent.agent_settings.max_speed
        # goal = self.way_points_queue[0] need goal points!!!!

        dynamic_window = self.calc_dynamic_window(self, vehicle_velo, vehicle_control, vehicle_transform, max_speed)
        #goal is an [x,y] position
        u, trajectory = self.calc_control_and_trajectory(dynamic_window, vehicle_velo, vehicle_control, vehicle_transform, goal, max_speed)  # need to find goal

        # for i in np.arange(len(trajectory) - 1):
        # control should just be a steering and longitude velocity
        next_waypoint = trajectory[:,-1]

        control = self.controller.run_in_series(next_waypoint=next_waypoint)
        return control

    def calc_dynamic_window(self, vehicle_velo, vehicle_control, vehicle_transform, max_speed):   #i think i need help calculating this
        # Dynamic window from robot specification, I think they cap the inputs to the throttle and steering at -1 and 1 respectively
        Vs = [0, max_speed,
              -1, 1]
        # Vs = [config.min_speed, config.max_speed,
        #       -config.max_yaw_rate, config.max_yaw_rate]
        # Dynamic window from motion model
        dt = 0.25

        # x = initial state [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
        Vd = [x[3] - config.max_accel * dt,
              x[3] + config.max_accel * dt,
              x[4] - config.max_delta_yaw_rate * dt,
              x[4] + config.max_delta_yaw_rate * dt]

        #  [v_min, v_max, yaw_rate_min, yaw_rate_max]
        dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]), max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]

        return dw

    def calc_control_and_trajectory(self, dw, vehicle_velo, vehicle_control, vehicle_transform, goal, max_speed):
        min_cost = np.inf
        best_u = [0.0, 0.0]

        throttle, steer = vehicle_control[0], vehicle_control[1]
        xvelo, yvelo, zvelo = vehicle_velo[0], vehicle_velo[1], vehicle_velo[2]
        x, y, z, roll, pitch, yaw = vehicle_transform[0], vehicle_transform[1], vehicle_transform[2], vehicle_transform[3], vehicle_transform[4], vehicle_transform[5]
        initial_state = [x, y, yaw, vehicle_velo, steer]  #check that yaw is in radians!!!!!!
        best_trajectory = np.array(initial_state)

        # x = initial state [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
        for v in np.arange(dw[0], dw[1], 0.01):
            for y in np.arange(dw[2], dw[3], 0.01):
                trajectory = self.predict_trajectory(initial_state, v, y)

                # calc cost
                #can change these variables to have quicker or safer ride
                to_goal_cost_gain = 0.15
                speed_cost_gain = 1.0
                obstacle_cost_gain = 1.0

                to_goal_cost = to_goal_cost_gain * self.calc_to_goal_cost(trajectory, goal)
                speed_cost = speed_cost_gain * (max_speed - trajectory[-1, 3])
                ob_cost = obstacle_cost_gain * self.calc_obstacle_cost(trajectory, obstacle)    #how to get obstacles

                final_cost = to_goal_cost + speed_cost + ob_cost

                # search for minimum cost
                if min_cost >= final_cost:
                    min_cost = final_cost
                    best_u = [v, y]
                    best_trajectory = trajectory

        return best_u, best_trajectory

    def calc_obstacle_cost(trajectory, obstacle):   #needs work
        """
        calc obstacle cost inf: collision
        """
        ox = obstacle[:, 0]
        oy = obstacle[:, 1]
        dx = trajectory[:, 0] - ox[:, None]
        dy = trajectory[:, 1] - oy[:, None]
        r = np.hypot(dx, dy)

        if config.robot_type == RobotType.rectangle:
            yaw = trajectory[:, 2]
            rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            rot = np.transpose(rot, [2, 0, 1])
            local_ob = ob[:, None] - trajectory[:, 0:2]
            local_ob = local_ob.reshape(-1, local_ob.shape[-1])
            local_ob = np.array([local_ob @ x for x in rot])
            local_ob = local_ob.reshape(-1, local_ob.shape[-1])
            upper_check = local_ob[:, 0] <= config.robot_length / 2
            right_check = local_ob[:, 1] <= config.robot_width / 2
            bottom_check = local_ob[:, 0] >= -config.robot_length / 2
            left_check = local_ob[:, 1] >= -config.robot_width / 2
            if (np.logical_and(np.logical_and(upper_check, right_check),
                               np.logical_and(bottom_check, left_check))).any():
                return float("Inf")
        elif config.robot_type == RobotType.circle:
            if np.array(r <= config.robot_radius).any():
                return float("Inf")

        min_r = np.min(r)
        return 1.0 / min_r  # OK

    def calc_to_goal_cost(trajectory, goal):
        """
            calc to goal cost with angle difference
        """

        dx = goal[0] - trajectory[-1, 0]
        dy = goal[1] - trajectory[-1, 1]
        error_angle = np.atan2(dy, dx)
        cost_angle = error_angle - trajectory[-1, 2]
        cost = abs(np.atan2(np.sin(cost_angle), np.cos(cost_angle)))

        return cost

    def predict_trajectory(initial_state, v, y):
        """
        predict trajectory with an input
        """

        def motion(x, u, dt):
            """
            motion model
            """

            x[2] += u[1] * dt
            x[0] += u[0] * np.cos(x[2]) * dt
            x[1] += u[0] * np.sin(x[2]) * dt
            x[3] = u[0]
            x[4] = u[1]

            return x

        x = np.array(initial_state)
        trajectory = np.array(x)
        time = 0
        while time <= 0.25:
            x = motion(x, [v, y], 0.01)
            trajectory = np.vstack((trajectory, x))
            time += 0.01

        return trajectory






