from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.base_env import BaseEnv, ActionTuple
from gym_unity.envs import UnityToGymWrapper
from PIL import Image, ImageDraw
import gym
import math
import numpy as np
import cv2 as cv
import io
from threading import Thread
import keyboard
import time
from Cogenvdecoder import  UdpComms as U




class CogEnvDecoder:
    def __init__(self, train=True, allow_multiple_obs=True, env_name='RealGame.exe', mat_num=0, no_graphics=False, time_scale=1):
        if train:
            engine_configuration_channel = EngineConfigurationChannel()
            engine_configuration_channel.set_configuration_parameters(width=100, height=100, time_scale=time_scale)
            unity_env = UnityEnvironment(env_name, worker_id=2, no_graphics=no_graphics)
            self._env = UnityToGymWrapper(unity_env)
        else:
            unity_env = UnityEnvironment(env_name, worker_id=1)
            self._env = UnityToGymWrapper(unity_env)

        self._size = (300,300)
        # print(self._env.reward_range)
        self.reward_range = self._env.reward_range
        self.metadata = self._env.metadata
        self.allow_multiple_obs = allow_multiple_obs
        self.spec = self._env.spec
        self._end_thread = False
        self._sock = U.UdpComms(udpIP="127.0.0.1", portTX=8000, portRX=8001, enableRX=True, suppressWarnings=True)
        self.depth_sock = U.UdpComms(udpIP="127.0.0.1", portTX=8002, portRX=8003,enableRX=True, suppressWarnings=True)
        self.last_frame = np.zeros((100, 100), dtype=np.uint8)
        self.last_depth_frame = np.zeros((100, 100), dtype=np.uint8)
        self._load_step = False #allow action transfer to run
        self._obs = None
        self._reward = None
        self._done = None
        self._info = None
        self._action = [0, 0, 0, 0]
        self._load = 0 #Check whether the action has been loaded
        if env_name == 'RealGame.exe':
            data = open("RealGame_Data/data.txt", "w")
            data.write(str(mat_num))
            data.close()
        elif env_name == '1.x86_64':
            data = open("data.txt", "w")
            data.write(str(mat_num))
            data.close()


    @property
    def observation_space(self):
        shape_image = self._size + (3,)
        color_image = gym.spaces.Box(low=0, high=255, shape=shape_image, dtype=np.uint8)
        num_vector  =26
        space_vector = gym.spaces.Box(low=-np.Inf, high=np.Inf, shape=(num_vector,), dtype=np.float32)
        depth_image = gym.spaces.Box(low=0, high=255, shape=shape_image, dtype=np.uint8)
        return gym.spaces.Dict({'color_image': space_image, 'depth_image': depthimage, 'vector': space_vector})
        # return space_image

    @property
    def action_space(self):
        return self._env.action_space

    def close(self):
        return self._env.close()

    def reset(self):
        self._end_thread = True
        # self._thread.join()
        obs = self._env.reset()
        self._end_thread = False
        # self.step_thread_begin()
        img = self.read_img()
        self_pos = self._GetSelfPos(obs)  # Blue one Position
        enemy_pos = self._EnemyPos(obs)  # Red one position
        goal1 = self._GetGoal1Pos(obs)  # Goal 1 position, whether it has been activated
        goal2 = self._GetGoal2Pos(obs)  # Goal 2 position, whether it has been activated
        goal3 = self._GetGoal3Pos(obs)  # Goal 3 position, whether it has been activated
        goal4 = self._GetGoal4Pos(obs)  # Goal 4 position, whether it has been activated
        goal5 = self._GetGoal5Pos(obs)  # Goal 5 position, whether it has been activated
        self_info = self._SelfInfo(obs)  # Blue one remaining HP & bullet
        collision_info = self._CollCondtion(obs)  # collision times and continuous collision time
        final_obs = [img, self_pos, enemy_pos, self_info, goal1, goal2, goal3, goal4, goal5, collision_info]
        return final_obs

    def action_transfer(self, action):
        action = np.array(action).reshape((1, 4))
        action_tuple = ActionTuple()
        action_tuple.add_continuous(action)
        self._env._env.set_actions(self._env.name, action_tuple)

        decision_step, terminal_step = self._env._env.get_steps(self._env.name)
        self._env._check_agents(max(len(decision_step), len(terminal_step)))
        #print(self._env.name)
        if len(terminal_step) != 0:
            # The agent is done
            self._env.game_over = True
            return self._env._single_step(terminal_step)
        else:
            return self._env._single_step(decision_step)


    def step_action(self, action):
        #obs, reward, done, info = self.action_transfer(action)
        obs, reward, done, info = self._env.step(action)
        #start = time.time()
        #self.action_transfer(action)
        #print(time.time() - start)
        #obs = self._obs
        #reward = self._reward
        #done = self._done
        #info = self._info
        # img = self.read_img()
        # depth_img = self.read_depth_img()
        # self_pos = self._GetSelfPos(obs)  # Blue one Position
        # enemy_pos = self._EnemyPos(obs)  # Red one position
        # goal1 = self._GetGoal1Pos(obs)  # Goal 1 position, whether it has been activated
        # goal2 = self._GetGoal2Pos(obs)  # Goal 2 position, whether it has been activated
        # goal3 = self._GetGoal3Pos(obs)  # Goal 3 position, whether it has been activated
        # goal4 = self._GetGoal4Pos(obs)  # Goal 4 position, whether it has been activated
        # goal5 = self._GetGoal5Pos(obs)  # Goal 5 position, whether it has been activated
        # flag_ach = self._AchievedGoals(obs)  # Num of goals have been achieved
        # enemy_act = self._EnemyStatus(obs)  # Whether enemy has been activated
        # enemy_info = self._EnemyInfo(obs)  # Red one remaining HP & bullets
        # score = self._Score(obs)  # Current score
        # self_info = self._SelfInfo(obs)  # Blue one remaining HP & bullet
        # dmg = self._DmgCaused(obs)  # blue one caused damage to red one
        # time_taken = self._TimeTaken(obs)  # Time passed in the round
        # collision_info = self._CollCondtion(obs)  # collision times and continuous collision time
        # final_obs = [img, self_pos, enemy_pos, self_info, goal1, goal2, goal3, goal4, goal5, collision_info, depth_img]
        # judge_result = [score, flag_ach, enemy_act, enemy_info, dmg, time_taken]
        img = self.read_img()
        depth_img = self.read_depth_img()
        self_pos = self._GetSelfPos(obs)  # Blue one Position
        enemy_pos = self._EnemyPos(obs)  # Red one position
        goal1 = self._GetGoal1Pos(obs)  # Goal 1 position, whether it has been activated
        goal2 = self._GetGoal2Pos(obs)  # Goal 2 position, whether it has been activated
        goal3 = self._GetGoal3Pos(obs)  # Goal 3 position, whether it has been activated
        goal4 = self._GetGoal4Pos(obs)  # Goal 4 position, whether it has been activated
        goal5 = self._GetGoal5Pos(obs)  # Goal 5 position, whether it has been activated
        flag_ach = self._AchievedGoals(obs)  # Num of goals have been achieved
        enemy_act = self._EnemyStatus(obs)  # Whether enemy has been activated
        enemy_info = self._EnemyInfo(obs)  # Red one remaining HP & bullets
        score = self._Score(obs)  # Current score
        self_info = self._SelfInfo(obs)  # Blue one remaining HP & bullet
        dmg = self._DmgCaused(obs)  # blue one caused damage to red one
        time_taken = self._TimeTaken(obs)  # Time passed in the round
        collision_info = self._CollCondtion(obs)  # collision times and continuous collision time
        final_obs = [img, self_pos, enemy_pos, self_info, goal1, goal2, goal3, goal4, goal5, collision_info,
                        depth_img]
        judge_result = [score, time_taken, collision_info, dmg, flag_ach, enemy_info, self_info]


        return final_obs, reward, done, [info, judge_result]

    def render(self, mode):
        return self._env.render(mode)

    def read_img(self):
        img = self._sock.ReadReceivedData()
        if not isinstance(img, np.ndarray):
            img = self.last_frame
        else:
            self.last_frame = img
        return img

    def read_depth_img(self):
        depth_img = self.depth_sock.ReadReceivedData()
        if not isinstance(depth_img, np.ndarray):
            depth_img = self.last_depth_frame
        else:
            self.last_depth_frame = depth_img
        return depth_img


    def _GetImg(self, imgData):
        img_length = int(imgData[100000])
        if img_length > 0:
            float_img = np.asarray(imgData[0:img_length], dtype=np.uint8)
            b = bytearray()
            for ele in float_img:
                b += ele.tobytes()
            byteio = io.BytesIO(b)
            img = Image.open(byteio)
            img2 = img.convert('RGB')
            opencv_img = np.array(img2)
            opencv_img = opencv_img[:, :, ::-1].copy()
            return opencv_img
        else:
            #print('The Environment has not finished loading')
            gray0 = np.zeros((300, 300), dtype=np.uint8)
            return gray0

    def _GetSelfPos(self, data):
        xPos = data[0]
        zPos = data[1]
        angle = data[2]
        return [xPos, zPos, angle]

    def _GetGoal1Pos(self, data):
        xPos = data[3]
        zPos = data[4]
        if data[13] >= 1:
            return [xPos, zPos, True]
        else:
            return [xPos, zPos, False]

    def _GetGoal2Pos(self, data):
        xPos = data[5]
        zPos = data[6]
        if data[13] >= 2:
            return [xPos, zPos, True]
        else:
            return [xPos, zPos, False]

    def _GetGoal3Pos(self, data):
        xPos = data[7]
        zPos = data[8]
        if data[13] >= 3:
            return [xPos, zPos, True]
        else:
            return [xPos, zPos, False]

    def _GetGoal4Pos(self, data):
        xPos = data[9]
        zPos = data[10]
        if data[13] >= 4:
            return [xPos, zPos, True]
        else:
            return [xPos, zPos, False]

    def _GetGoal5Pos(self, data):
        xPos = data[11]
        zPos = data[12]
        if data[13] > 4:
            return [xPos, zPos, True]
        else:
            return [xPos, zPos, False]

    def _AchievedGoals(self, data):
        return data[13]

    def _EnemyStatus(self, data):
        return (data[13] == 5)

    def _EnemyPos(self, data):
        xPos = data[14]
        zPos = data[15]
        angle = data[16]
        return [xPos, zPos, angle]

    def _EnemyInfo(self, data):
        HP = data[17]
        Bullet = data[18]
        return [HP, Bullet]

    def _Score(self, data):
        return data[19]

    def _SelfInfo(self, data):
        HP = data[20]
        Bullet = data[25]
        return [HP, Bullet]

    def _DmgCaused(self, data):
        dmg = data[21]
        return dmg

    def _TimeTaken(self, data):
        time = data[22]
        return time

    def _CollCondtion(self, data):
        coll_time = data[23]
        cont_coll_time = data[24]
        return [coll_time, cont_coll_time]

    def _thread_step(self):
        quit = False
        #print('1')
        while True:
            self._env._env.step()
            time.sleep(0.02)
            if self._end_thread == True:
                return

    def _thread_step_action(self):
        done = False
        while done == False and self._end_thread == False:
            #print(done)
            #if np.count_nonzero(np.asarray(self._action)) != 4 and self._load == 0:
            #    self._load = 1
                #print('here')

            obs, reward, done, info = self._env.step(self._action)
            #if self._load == 1:
                #print(self._action)
            #    self._action = [0, 0, 0, 0]
            #print(self._action)
            img = self.read_img()
            depth_img = self.read_depth_img()
            self_pos = self._GetSelfPos(obs)  # Blue one Position
            enemy_pos = self._EnemyPos(obs)  # Red one position
            goal1 = self._GetGoal1Pos(obs)  # Goal 1 position, whether it has been activated
            goal2 = self._GetGoal2Pos(obs)  # Goal 2 position, whether it has been activated
            goal3 = self._GetGoal3Pos(obs)  # Goal 3 position, whether it has been activated
            goal4 = self._GetGoal4Pos(obs)  # Goal 4 position, whether it has been activated
            goal5 = self._GetGoal5Pos(obs)  # Goal 5 position, whether it has been activated
            flag_ach = self._AchievedGoals(obs)  # Num of goals have been achieved
            enemy_act = self._EnemyStatus(obs)  # Whether enemy has been activated
            enemy_info = self._EnemyInfo(obs)  # Red one remaining HP & bullets
            score = self._Score(obs)  # Current score
            self_info = self._SelfInfo(obs)  # Blue one remaining HP & bullet
            dmg = self._DmgCaused(obs)  # blue one caused damage to red one
            time_taken = self._TimeTaken(obs)  # Time passed in the round
            collision_info = self._CollCondtion(obs)  # collision times and continuous collision time
            final_obs = [img, self_pos, enemy_pos, self_info, goal1, goal2, goal3, goal4, goal5, collision_info,
                         depth_img]
            judge_result = [score, time_taken, collision_info, dmg, flag_ach, enemy_info, self_info]
            self._obs = final_obs
            self._reward = reward
            self._done = done
            self._info = [info, judge_result]




    def step_thread_begin(self):
        #self._thread = Thread(target=self._thread_step)
        self._thread = Thread(target=self._thread_step_action)

        self._thread.start()
        #print('3')
        #self._thread.join()
        #print('4')

    def set_action(self, action):
        self._action = action
        #self._load = 0
        return self._obs, self._reward, self._done, self._info
