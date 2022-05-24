import numpy as np

class Agent:
    def __init__(self, model_path=None):
        self.model_path  = model_path
        # you can customize the necessary attributes here
    
    def agent_control(self, obs, done, info):
        # The formats of obs, done, info obey the  CogEnvDecoder api
        # realize your agent here
        action = [0.0, 0.0, 0.0, 0.0]

        # here is a simple demo
        action = self.simple_demo_control(obs, done, info)
        # return action:[vel_x, vel_y, vel_w, shoud_shoot]
        return action

    
    # -------------------- the following codes are for simple demo, can be deleted -----------------------
    def simple_demo_control(self, obs, done, info=None):
        action = [0.0, 0.0, 0.0, 0.0]
        vector_data = obs["vector"]
        num_activated_goals = 0
        if info is not None:
            num_activated_goals = info[1][3]
        self_pose = vector_data[0]
        enemy_activated = vector_data[2]
        enemy_pose = vector_data[3]
        goals_list = [vector_data[i] for i in range(5, 10)]

        if not enemy_activated:
            for goal in goals_list:
                is_activated = goal[-1]
                if is_activated:
                    continue
                else:
                    move_control = self.calculate_move_control(self_pose, goal[:2])
                    move_control.append(0.0) # add shoot control
                    action = move_control
                    break
        else:
            move_control = self.calculate_move_control(self_pose, enemy_pose[:2])
            dist_to_enemy = np.sqrt((self_pose[0]-enemy_pose[0])**2 + (self_pose[1]-enemy_pose[1])**2)
            if dist_to_enemy < 1.5:
                move_control[0] = np.random.uniform(-0.5, 0.5)
                move_control[1] = np.random.uniform(-0.5, 0.5)
            move_control.append(1.0)
            action = move_control
        return action

    def calculate_move_control(self, self_pose, target_position):
        delta_x = target_position[0] - self_pose[0]
        delta_y = target_position[1] - self_pose[1]
        x_in_robot = delta_x * np.cos(self_pose[2]) + delta_y * np.sin(self_pose[2])
        y_in_robot = -delta_x * np.sin(self_pose[2]) + delta_y * np.cos(self_pose[2])
        theta_in_robot = np.arctan2(y_in_robot, x_in_robot)

        vel_x = 1.0 * x_in_robot
        vel_y = 1.0 * y_in_robot
        vel_w = theta_in_robot
        
        return [vel_x, vel_y, vel_w]
    # --------------------------------------------------------------------------------------------