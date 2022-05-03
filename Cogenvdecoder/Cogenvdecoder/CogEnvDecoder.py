from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.base_env import BaseEnv, ActionTuple
from gym_unity.envs import UnityToGymWrapper
import gym
import math
import numpy as np
import cv2 as cv
import io

class CogEnvDecoder:
    def __init__(self, worker_id=1, train=True, allow_multiple_obs=True, env_name="linux_V1/1.x86_64", mat_num=0, no_graphics=False, time_scale=1):
        if train:
            engine_configuration_channel = EngineConfigurationChannel()
            engine_configuration_channel.set_configuration_parameters(width=50, height=50, time_scale=time_scale)
            unity_env = UnityEnvironment(env_name, worker_id=worker_id, no_graphics=no_graphics)
            self._env = UnityToGymWrapper(unity_env, allow_multiple_obs=True, uint8_visual=True)
        else:
            unity_env = UnityEnvironment(env_name, worker_id=worker_id)
            self._env = UnityToGymWrapper(unity_env, allow_multiple_obs=True, uint8_visual=True)

        self._size = (300,300)
        self.reward_range = self._env.reward_range
        self.metadata = self._env.metadata
        self.allow_multiple_obs = allow_multiple_obs
        self.spec = self._env.spec

        # self.last_frame = np.zeros((100, 100), dtype=np.uint8)
        # self.last_depth_frame = np.zeros((100, 100), dtype=np.uint8)
        self._load_step = False #allow action transfer to run
        self._obs = None
        self._reward = None
        self._done = None
        self._info = None
        self._action = [0, 0, 0, 0]
        self._load = 0 #Check whether the action has been loaded


    @property
    def observation_space(self):
        shape_image = self._size + (3,)
        color_image = gym.spaces.Box(low=0, high=255, shape=shape_image, dtype=np.uint8)
        num_laser = 61
        laser_space = gym.spaces.Box(low=0, high=100, shape=(num_laser,), dtype=np.float32)
        num_vector = 28
        space_vector = gym.spaces.Box(low=-np.Inf, high=np.Inf, shape=(num_vector,), dtype=np.float32)

        return gym.spaces.Dict({'color_image': color_image, 'laser': laser_space, 'vector': space_vector})

    @property
    def action_space(self):
        return self._env.action_space

    def close(self):
        return self._env.close()

    def reset(self):
  

        obs = self._env.reset()

        img = self._GetImg(obs)
        
        self_pos = self._GetSelfPos(obs)  # Blue one Position
        self_info = self._SelfInfo(obs)  # Blue one remaining HP & bullet

        
        goal1 = self._GetGoal1Pos(obs)  # Goal 1 position, whether it has been activated
        goal2 = self._GetGoal2Pos(obs)  # Goal 2 position, whether it has been activated
        goal3 = self._GetGoal3Pos(obs)  # Goal 3 position, whether it has been activated
        goal4 = self._GetGoal4Pos(obs)  # Goal 4 position, whether it has been activated
        goal5 = self._GetGoal5Pos(obs)  # Goal 5 position, whether it has been activated

        collision_info = self._CollCondtion(obs)  # collision times and continuous collision time

        enemy_act = self._EnemyStatus(obs)  # Whether enemy has been activated
        enemy_pos = self._EnemyPos(obs)  # Red one position
        enemy_info = self._EnemyInfo(obs)  # Red one remaining HP & bullets

        laser = self._GetLaser(obs)

        # [3, 2, 1, 3, 2, 3, 3, 3, 3, 3, 2]
        vector_state = [self_pos, self_info, enemy_act, enemy_pos, enemy_info, 
                        goal1, goal2, goal3, goal4, goal5, collision_info]
        state = {"color_image":img, "laser": laser, "vector": vector_state}
        return state


    def step(self, action):
        

        obs, reward, done, info = self._env.step(action)
        
        img = self._GetImg(obs)
 
        self_pos = self._GetSelfPos(obs)  # Blue one Position
        self_info = self._SelfInfo(obs)  # Blue one remaining HP & bullet

        goal1 = self._GetGoal1Pos(obs)  # Goal 1 position, whether it has been activated
        goal2 = self._GetGoal2Pos(obs)  # Goal 2 position, whether it has been activated
        goal3 = self._GetGoal3Pos(obs)  # Goal 3 position, whether it has been activated
        goal4 = self._GetGoal4Pos(obs)  # Goal 4 position, whether it has been activated
        goal5 = self._GetGoal5Pos(obs)  # Goal 5 position, whether it has been activated
        flag_ach = self._AchievedGoals(obs)  # Num of goals have been achieved
        collision_info = self._CollCondtion(obs)  # collision times and continuous collision time

        enemy_act = self._EnemyStatus(obs)  # Whether enemy has been activated
        enemy_pos = self._EnemyPos(obs)  # Red one position
        enemy_info = self._EnemyInfo(obs)  # Red one remaining HP & bullets
        
        laser = self._GetLaser(obs)

        score = self._Score(obs)  # Current score
        dmg = self._DmgCaused(obs)  # blue one caused damage to red one
        time_taken = self._TimeTaken(obs)  # Time passed in the round
        

        judge_result = [score, time_taken, dmg, flag_ach]

        vector_state = [self_pos, self_info, enemy_act, enemy_pos, enemy_info, 
                        goal1, goal2, goal3, goal4, goal5, collision_info]

        state = {"color_image":img, "laser": laser, "vector": vector_state}


        return state, reward, done, [info, judge_result]
    
    def check_state(self, state, info=None):
        image_data = state["color_image"]
        laser_data = np.array(state["laser"])
        vector_data = state["vector"]
        print("=======================state check====================")
        print("image shape: {}".format(image_data.shape))
        print("laser shape: {}, max distance: {}, min distance: {}".format(laser_data.shape, np.max(laser_data), np.min(laser_data)))
        print("self pose: {}, self info: {}, enemy active: {}, enemy pose: {}, enemy_info: {}".format(vector_data[0], vector_data[1], vector_data[2], vector_data[3], vector_data[4]))
        print("goal 1: {}, goal 2: {}, goal 3: {}, goal 4: {}, goal 5:{}".format(vector_data[5], vector_data[6], vector_data[7], vector_data[8], vector_data[9]))
        print("total collisions: {}, total collision time: {} ".format(vector_data[10][0], vector_data[10][1]))
        if info is not None:
            print("Number of goals have been activated: {}".format(info[1][3]))
            print("time taken: {}, attack damage: {}, score: {}".format(info[1][1], info[1][2], info[1][0]))
        print("-----------------------end check---------------------")
    def render(self, mode):
        return self._env.render(mode)


    def _GetImg(self, data):
        return cv.cvtColor(data[0], cv.COLOR_RGB2BGR)

    def _GetSelfPos(self, data):
        xPos = data[1][0]
        zPos = data[1][1]
        angle = data[1][2]
        return [xPos, zPos, angle]

    def _GetGoal1Pos(self, data):
        xPos = data[1][3]
        zPos = data[1][4]
        if data[1][13] >= 1:
            return [xPos, zPos, True]
        else:
            return [xPos, zPos, False]

    def _GetGoal2Pos(self, data):
        xPos = data[1][5]
        zPos = data[1][6]
        if data[1][13] >= 2:
            return [xPos, zPos, True]
        else:
            return [xPos, zPos, False]

    def _GetGoal3Pos(self, data):
        xPos = data[1][7]
        zPos = data[1][8]
        if data[1][13] >= 3:
            return [xPos, zPos, True]
        else:
            return [xPos, zPos, False]

    def _GetGoal4Pos(self, data):
        xPos = data[1][9]
        zPos = data[1][10]
        if data[1][13] >= 4:
            return [xPos, zPos, True]
        else:
            return [xPos, zPos, False]

    def _GetGoal5Pos(self, data):
        xPos = data[1][11]
        zPos = data[1][12]
        if data[1][13] > 4:
            return [xPos, zPos, True]
        else:
            return [xPos, zPos, False]

    def _AchievedGoals(self, data):
        return data[1][13]

    def _EnemyStatus(self, data):
        return (data[1][13] == 5)

    def _EnemyPos(self, data):
        xPos = data[1][14]
        zPos = data[1][15]
        angle = data[1][16]
        return [xPos, zPos, angle]

    def _EnemyInfo(self, data):
        HP = data[1][17]
        Bullet = data[1][18]
        return [HP, Bullet]

    def _Score(self, data):
        return data[1][19]

    def _SelfInfo(self, data):
        HP = data[1][20]
        Bullet = data[1][25]
        return [HP, Bullet]

    def _DmgCaused(self, data):
        dmg = data[1][21]
        return dmg

    def _TimeTaken(self, data):
        time = data[1][22]
        return time

    def _CollCondtion(self, data):
        coll_time = data[1][23]
        cont_coll_time = data[1][24]
        return [coll_time, cont_coll_time]

    def _GetLaser(self, data):
        laser_data = data[1][26:]
        return laser_data
    
