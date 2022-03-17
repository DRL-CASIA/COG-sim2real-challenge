from time import sleep
import numpy as np
import math

class TrapzoidPlanner():
    def __init__(self, max_v=3.0, max_a=3.0, target_v=1.0):
        self.current_vx = 0.0
        self.current_vy = 0.0
        self.current_vw = 0.0

        self.current_v = 0.0
        self.max_v = max_v
        self.max_vw = 3.0
        self.max_a = max_a

        self.target_v = target_v

        self.path = None
    
    def set_path(self, path):
        self.path = path
    
    def set_current_velocity(self, vx, vy):
        self.current_vx = vx
        self.current_vy = vy
        # self.current_vw = vw
        self.current_v = math.sqrt((self.current_vx)**2+(self.current_vy)**2)

    
    def calc_distance(self, p1, p2):
        return math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)
    
    def calc_path_length(self):
        if self.path is None:
            print("Path is not initialize!")
        else:
            if len(self.path)>1:
                total_dist = 0.0
                for i in range(1, len(self.path)):
                    total_dist += self.calc_distance(self.path[i], self.path[i-1])
                return total_dist
            else:
                print("Path length is too short!")
                return 0.0
        return 0.0
    
    def calc_max_velocity(self):
        s1 = ((self.max_v)**2-(self.current_v)**2)/(2 * self.max_a)
        s3 = ((self.max_v)**2-(self.target_v)**2)/(2 * self.max_a)

        s = self.calc_path_length()
        s2 = s - s1 - s3
        # print("init: ", s1, s2, s3)
        if s2>0:
            return self.max_v, s1, s2, s3
        else:
            max_v = math.sqrt((2*self.max_a*s + (self.current_v)**2 + (self.target_v)**2)/2.0)
            s1 = ((max_v)**2-(self.current_v)**2)/(2 * self.max_a)
            s3 = ((max_v)**2-(self.target_v)**2)/(2 * self.max_a)
            # print(s1, s3)
            s3 = s - s1
            s2 = 0.0
            return max_v, s1, s2, s3
    
    def plan(self, path):
        self.set_path(path)
        max_v, s1, s2, s3 = self.calc_max_velocity()
        trajectory = []
        pass_dist = 0.0
        cur_v = self.current_v
        # print("final: ",s1, s2, s3)
        for i in range(len(path)):
            if i == 0:
                cur_v = self.current_v
                px, py, vx, vy = path[i][0], path[i][1], self.current_vx, self.current_vy
                # trajectory.append([px, py, vx, vy])
                trajectory.append([px, py, self.current_v])
            else:
                # print(path[i], path[i-1])
                dist = self.calc_distance(path[i], path[i-1])

                pass_dist += dist
                # print(pass_dist)
                if pass_dist < s1:
                    cur_v = math.sqrt(cur_v**2+2*self.max_a*dist)
                    
                elif pass_dist < (s1+s2):
                    cur_v = max_v
                    
                else:
                    # print(cur_v**2, 2*self.max_a*dist)
                    if cur_v**2-2*self.max_a*dist < 0 or cur_v<self.target_v:
                        # print("cur_v:",cur_v," max_a:",self.max_a," dist:", dist)
                        cur_v = cur_v * 0.9
                    else:
                        cur_v =  math.sqrt(cur_v**2-2*self.max_a*dist)
                        # print("cur_v:",cur_v)
                theta = math.atan2((path[i][1]-path[i-1][1]), (path[i][0]-path[i-1][0]))
                cur_vx = cur_v * math.cos(theta)
                cur_vy = cur_v * math.sin(theta)
                # trajectory.append([path[i][0], path[i][1], cur_vx, cur_vy])
                trajectory.append([path[i][0], path[i][1], cur_v])
        
        return trajectory
    

class Controller():
    def __init__(self):
        self.robot_vx = 0.0
        self.robot_vy = 0.0
        self.robot_vw = 0.0
        self.robot_px = 0.0
        self.robot_py = 0.0
        self.robot_pw = 0.0
        self.robot_v = 0.0

        self.Kp = 0.8
        self.Kd = 0.2
    
    def set_robot_state(self, vx, vy, vw, px, py, pw):
        self.robot_vx = vx
        self.robot_vy = vy
        self.robot_vw = vw
        self.robot_px = px
        self.robot_py = py
        self.robot_pw = pw
        self.robot_v = math.sqrt(vx**2+vy**2)
    
    # def PIDController(self, target_vx, target_vy, target_px, target_py):
    #     # robot coord -> world coord
    #     # robot_to_world_mat = np.array([[np.cos(self.robot_pw), -np.sin(self.robot_pw)],
    #     #                                [np.sin(self.robot_pw),  np.cos(self.robot_pw)]])
    #     world_vx = self.robot_vx * math.cos(self.robot_pw) - self.robot_vy * math.sin(self.robot_pw)
    #     world_vy = self.robot_vx * math.sin(self.robot_pw) + self.robot_vy * math.cos(self.robot_pw)

    #     world_control_vx = self.Kp * (target_px - self.robot_px) + self.Kd * (target_vx)
    #     world_control_vy = self.Kp * (target_py - self.robot_py) + self.Kd * (target_vy)

    #     control_vx = world_control_vx * math.cos(self.robot_pw) + world_control_vy * math.sin(self.robot_pw)
    #     control_vy = -world_control_vx * math.sin(self.robot_pw) + world_control_vy * math.cos(self.robot_pw)
    #     return control_vx, control_vy
    
    def PIDController(self, target_v, target_px, target_py, dt=0.1):
        # robot coord -> world coord
        # robot_to_world_mat = np.array([[np.cos(self.robot_pw), -np.sin(self.robot_pw)],
        #                                [np.sin(self.robot_pw),  np.cos(self.robot_pw)]])
        delta_x = target_px - self.robot_px
        delta_y = target_py - self.robot_py

        control_a = self.Kp * (target_v - self.robot_v)
        world_vx = self.robot_vx * math.cos(self.robot_pw) - self.robot_vy * math.sin(self.robot_pw)
        world_vy = self.robot_vx * math.sin(self.robot_pw) + self.robot_vy * math.cos(self.robot_pw)

        world_control_vx = self.Kp * (target_px - self.robot_px) + self.Kd * (target_vx)
        world_control_vy = self.Kp * (target_py - self.robot_py) + self.Kd * (target_vy)

        control_vx = world_control_vx * math.cos(self.robot_pw) + world_control_vy * math.sin(self.robot_pw)
        control_vy = -world_control_vx * math.sin(self.robot_pw) + world_control_vy * math.cos(self.robot_pw)
        return control_vx, control_vy
    
