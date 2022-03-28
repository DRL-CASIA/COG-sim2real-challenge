from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from gym_unity.envs import UnityToGymWrapper
from PIL import Image, ImageDraw
import gym
import math
import numpy as np
import cv2 as cv
import io
num_vector = 7

class CogEnvDecoder:
    def __init__(self, size=(400, 400), train=True, allow_multiple_obs=True, env_name='RealGame.exe'):
        if train:
            engine_configuration_channel = EngineConfigurationChannel()
            engine_configuration_channel.set_configuration_parameters(width=100, height=100, time_scale=100)
            unity_env = UnityEnvironment(env_name, worker_id=2)
            self._env = UnityToGymWrapper(unity_env)
        else:
            unity_env = UnityEnvironment(env_name, worker_id=1)
            self._env = UnityToGymWrapper(unity_env)

        self._size = size
        # print(self._env.reward_range)
        self.reward_range = self._env.reward_range
        self.metadata = self._env.metadata
        self.allow_multiple_obs = allow_multiple_obs
        self.spec = self._env.spec 
        # print("init")
    
    @property
    def observation_space(self):
        shape_image = self._size + (3,)
        space_image = gym.spaces.Box(low=0, high=255, shape=shape_image, dtype = np.uint8)
        space_vector = gym.spaces.Box(low=-np.Inf, high=np.Inf, shape=(num_vector,), dtype = np.float32)
        return gym.spaces.Dict({'image':space_image, 'vector':space_vector})
        # return space_image

    @property
    def action_space(self):
        return self._env.action_space
    
    def close(self):
        return self._env.close()

    def reset(self):
        obs = self._env.reset()
        img = self._GetImg(obs)
        self_pos = self._GetSelfPos(obs) #Blue one Position
        enemy_pos = self._EnemyPos(obs)#Red one position
        goal1 = self._GetGoal1Pos(obs)#Goal 1 position, whether it has been activated
        goal2 = self._GetGoal2Pos(obs)#Goal 2 position, whether it has been activated
        goal3 = self._GetGoal3Pos(obs)#Goal 3 position, whether it has been activated
        goal4 = self._GetGoal4Pos(obs)#Goal 4 position, whether it has been activated
        goal5 = self._GetGoal5Pos(obs)#Goal 5 position, whether it has been activated
        self_info = self._SelfInfo(obs)#Blue one remaining HP & bullet
        final_obs = [img, self_pos, enemy_pos, self_info, goal1, goal2, goal3, goal4, goal5]
        return final_obs

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        #obs = self._get_obs(obs)
        img = self._GetImg(obs)
        self_pos = self._GetSelfPos(obs) #Blue one Position
        enemy_pos = self._EnemyPos(obs)#Red one position
        goal1 = self._GetGoal1Pos(obs)#Goal 1 position, whether it has been activated
        goal2 = self._GetGoal2Pos(obs)#Goal 2 position, whether it has been activated
        goal3 = self._GetGoal3Pos(obs)#Goal 3 position, whether it has been activated
        goal4 = self._GetGoal4Pos(obs)#Goal 4 position, whether it has been activated
        goal5 = self._GetGoal5Pos(obs)#Goal 5 position, whether it has been activated
        flag_ach = self._AchievedGoals(obs)#Num of goals have been achieved
        enemy_act = self._EnemyStatus(obs)#Whether enemy has been activated
        enemy_info = self._EnemyInfo(obs)#Red one remaining HP & bullets
        score = self._Score(obs)#Current score
        self_info = self._SelfInfo(obs)#Blue one remaining HP & bullet
        dmg = self._DmgCaused(obs)#blue one caused damage to red one
        time = self._TimeTaken(obs)#Time passed in the round
        collision_info = self._CollCondtion(obs)#collision times and continuous collision time
        final_obs = [img, self_pos, enemy_pos, self_info, goal1, goal2, goal3, goal4, goal5]
        judge_result = [score, flag_ach, enemy_act, enemy_info, dmg, time, collision_info]
        return final_obs, reward, done, [info, judge_result]
    
    def render(self, mode):
        return self._env.render(mode)

    def _get_obs(self, obs):
        img = np.zeros((400, 400, 3), np.uint8)
        # end_position = (math.floor(obs[0]*100),math.floor(obs[1]*100))
        # print("我是end_position:",end_position)
        # cv.circle(img, end_position, 2, (0, 0, 255), 4)#BGR
        # ptStart = (math.floor(obs[2]*100),math.floor(obs[3]*100))
        ptStart = (200, 200)
        # print("我是ptStart:",ptStart)
        robot_angle = obs[2]
        yaw = 0
        test_yaw = []
        vector = []
        vector.append(obs[0])
        vector.append(obs[1])
        vector.append(obs[2])
        vector.append(obs[3])
        vector.append(obs[4])
        vector.append(obs[5])
        vector.append(obs[6])
        for i in range(len(obs)):
            if i < num_vector:
                continue
            elif i > 6 and i < 120:
                yaw = i - num_vector
                # yaw = 0
                
            else:
                yaw = i + 130
                # yaw = 0
            
            late_yaw = 90/57.3 - (robot_angle - yaw/57.3)
            # test_yaw.append(late_yaw)
            # print(i)
            # print(len(test_yaw))
            distance = obs[i]
            ptEnd = (math.floor(distance*math.cos(late_yaw)*100)+200, math.floor(distance*math.sin(late_yaw)*100)+200)
            point_color = (0, 255, 0)
            thickness = 1
            lineType = 4 
            cv.line(img, ptStart, ptEnd, point_color, thickness, lineType)        
        # cv.namedWindow("image")
        img = cv.flip(img,0)
        # print(len(test_yaw))
        # cv.imshow('image', img)
        # cv.waitKey(600)
        # cv.destroyAllWindows()
        return {'image':img, 'vector':np.array(vector)}
        # return img


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
            print('The Environment has not finished loading')
            gray0=np.zeros((300,300),dtype=np.uint8)
            return gray0

    def _GetSelfPos(self, data):
        xPos = data[100001]
        zPos = data[100002]
        angle = data[100003]
        return [xPos, zPos, angle]


    def _GetGoal1Pos(self, data):
        xPos = data[100004]
        zPos = data[100005]
        if data[100014] >= 1:
            return [xPos, zPos, True]
        else:
            return [xPos,zPos,False]


    def _GetGoal2Pos(self, data):
        xPos = data[100006]
        zPos = data[100007]
        if data[100014] >= 2:
            return [xPos, zPos, True]
        else:
            return [xPos,zPos,False]


    def _GetGoal3Pos(self, data):
        xPos = data[100008]
        zPos = data[100009]
        if data[100014] >= 3:
            return [xPos, zPos, True]
        else:
            return [xPos,zPos,False]


    def _GetGoal4Pos(self, data):
        xPos = data[100010]
        zPos = data[100011]
        if data[100014] >= 4:
            return [xPos, zPos, True]
        else:
            return [xPos,zPos,False]

    def _GetGoal5Pos(self, data):
        xPos = data[100012]
        zPos = data[100013]
        if data[100014] > 4:
            return [xPos, zPos, True]
        else:
            return [xPos,zPos,False]


    def _AchievedGoals(self, data):
        return data[10014]


    def _EnemyStatus(self, data):
        return (data[100014] == 5)


    def _EnemyPos(self, data):
        xPos = data[100015]
        zPos = data[100016]
        angle = data[100017]
        return [xPos, zPos, angle]


    def _EnemyInfo(self, data):
        HP = data[100018]
        Bullet = data[100019]
        return [HP, Bullet]


    def _Score(self, data):
        return data[100020]

    def _SelfInfo(self, data):
        HP = data[100021]
        Bullet = data[100026]
        return [HP, Bullet]

    def _DmgCaused(self, data):
        dmg = data[100022]
        return dmg


    def _TimeTaken(self, data):
        time = data[100023]
        return time


    def _CollCondtion(self, data):
        coll_time = data[100024]
        cont_coll_time = data[100025]
        return [coll_time, cont_coll_time]
