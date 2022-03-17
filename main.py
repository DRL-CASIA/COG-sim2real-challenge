import math
from pickle import NONE
from turtle import goto
import matplotlib.pyplot as plt
import numpy as np
import cv2 
import time
from planning.bidirectional_a_star_costmap import BidirectionalAStarCostmapPlanner
# from mlagents_envs.environment import UnityEnvironment
# from gym_unity.envs import UnityToGymWrapper
from PIL import Image
import io
from identify.apriltag_s import Apriltag
from identify import tagUtils as tud
import random
show_animation = True
from planning.speed_planner import TrapzoidPlanner
from Cogenvdecoder import CogEnvDecoder

# Parameters
k = 0.1  # look forward gain
Lfc = 0.25  # [m] look-ahead distance
Kp = 1.0  # speed proportional gain
dt = 0.1  # [s] time tick
WB = 2.9  # [m] wheel base of vehicle


class State:

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
    
    def update_mec(self, a, delta):
        self.x += self.v * math.cos(delta+self.yaw) * dt
        self.y += self.v * math.sin(delta+self.yaw) * dt
        if delta > np.pi:
            delta -= 2* np.pi
        if delta < -np.pi:
            delta += 2* np.pi
        self.yaw += delta * dt
        self.v += a * dt

    def calc_distance(self, point_x, point_y):
        dx = self.x - point_x
        dy = self.y - point_y
        # print("dx:",dx,"dy:",dy)
        return math.hypot(dx, dy)

    def calc_control(self, a, delta, reverse=False, tx=None, ty=None):
        if reverse:
            delta = delta + np.pi
            if delta>np.pi:
                delta -= 2*np.pi
            if delta<-np.pi:
                delta += 2*np.pi
            vx = -(self.v + a) * math.cos(delta)
            vy = -(self.v + a) * math.sin(delta)
            vw = delta * 1.0
        else:
            if delta>np.pi:
                delta -= 2*np.pi
            if delta<-np.pi:
                delta += 2*np.pi
            vx = (self.v + a) * math.cos(delta)
            vy = (self.v + a) * math.sin(delta)
            vw = delta * 1.0

        if tx is not None and ty is not None:
            delta = math.atan2(ty - self.y, tx - self.x) - self.yaw
            if delta>np.pi:
                delta -= 2*np.pi
            if delta<-np.pi:
                delta += 2*np.pi
            vw = delta * 3.0
        return vx, vy, vw

    def update_state(self, x, y, yaw):
        self.x = x
        self.y = y
        self.yaw = yaw

class States:

    def __init__(self):
        self.x = []
        self.y = []
        self.yaw = []
        self.v = []
        self.t = []

    def append(self, t, state):
        self.x.append(state.x)
        self.y.append(state.y)
        self.yaw.append(state.yaw)
        self.v.append(state.v)
        self.t.append(t)

def proportional_control(target, current):
    a = Kp * (target - current)

    return a


class TargetCourse:

    def __init__(self, cx, cy):
        self.cx = cx
        self.cy = cy
        self.old_nearest_point_index = None

    def search_target_index(self, state):

        # To speed up nearest point search, doing it at only first time.
        if self.old_nearest_point_index is None:
            # search nearest point index
            dx = [state.x - icx for icx in self.cx]
            dy = [state.y - icy for icy in self.cy]
            d = np.hypot(dx, dy)
            ind = np.argmin(d)
            self.old_nearest_point_index = ind
        else:
            ind = self.old_nearest_point_index
            distance_this_index = state.calc_distance(self.cx[ind],
                                                      self.cy[ind])
            while True:
                if ind >= len(self.cx)-1:
                    break
                distance_next_index = state.calc_distance(self.cx[ind + 1],
                                                          self.cy[ind + 1])
                if distance_this_index < distance_next_index:
                    break
                ind = ind + 1 if (ind + 1) < len(self.cx) else ind
                distance_this_index = distance_next_index
            self.old_nearest_point_index = ind

        Lf = k * state.v + Lfc  
        while Lf > state.calc_distance(self.cx[ind], self.cy[ind]):
            if (ind + 1) >= len(self.cx):
                break  # not exceed goal
            ind += 1

        return ind, Lf

def pure_pursuit_control(state, trajectory, pind):
    ind, Lf = trajectory.search_target_index(state)

    if pind >= ind:
        ind = pind

    if ind < len(trajectory.cx):
        tx = trajectory.cx[ind]
        ty = trajectory.cy[ind]
    else:  # toward goal
        tx = trajectory.cx[-1]
        ty = trajectory.cy[-1]
        ind = len(trajectory.cx) - 1
    alpha = math.atan2(ty - state.y, tx - state.x) - state.yaw

    return alpha, ind


def plot_arrow(x, y, yaw, length=0.3, width=0.1, fc="r", ec="k"):
    """
    Plot arrow
    """

    if not isinstance(x, float):
        for ix, iy, iyaw in zip(x, y, yaw):
            plot_arrow(ix, iy, iyaw)
    else:
        plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
                  fc=fc, ec=ec, head_width=width, head_length=width)
        plt.plot(x, y)
def main(sx=1.0, sy=4.0, gx=7.0, gy=1.0, observation=None):

    # start and goal position
    sx = sx * 25 # [m]
    sy = sy * 25 # [m]
    gx = gx * 25 # [m]
    gy = gy * 25 # [m]
    obs = []
    for i in range(4,9,1):
        obs.append(observation[i][0] * 25)
        obs.append(observation[i][1] * 25)
    obs.append(observation[2][0] * 25)
    obs.append(observation[2][1] * 25)
    # obs.append(observation[13] * 25)
    # obs.append(observation[14] * 25)
    
    # set obstacle positions
    ox, oy = [], []
    img = cv2.imread('icra2021.pgm', 0)
    columes, rows = img.shape  # 448 808
    img_new = img.copy()
    for i in range(columes):
      for j in range(rows):
        if img_new[i, j] > 128:
            img_new[i, j] = 255
        else:
            img_new[i, j] = 0
            ox.append(j/4.)
            oy.append((columes-i)/4.)
    for j in range(0,12,2):
        for ix in range(int(obs[j])-4, int(obs[j])+5, 1):
            for iy in range(int(obs[j+1])-4, int(obs[j+1])+5, 1):
                ox.append(ix)
                oy.append(iy)
    
    if show_animation:  # pragma: no cover
        plt.plot(ox, oy, ".k")
        plt.plot(sx, sy, "og")
        plt.plot(gx, gy, "ob")
        plt.grid(True)
        plt.axis("equal")

    costmap = cv2.imread('costmap_25_11.png', 0)
    columes, rows = costmap.shape  # 112 202
    costmap = cv2.flip(costmap, 0, costmap)
    columes, rows = costmap.shape  # 112 202
    for j in range(0,12,2):
        for ix in range(int(obs[j])-6, int(obs[j])+7, 1):
            for iy in range(int(obs[j+1])-6, int(obs[j+1])+7, 1):
                costmap[iy][ix] = 0

    # if show_animation:  # pragma: no cover
    #     plt.plot(tempx, tempy, ".k")
    #     plt.plot(sx, sy, "og")
    #     plt.plot(gx, gy, "ob")
    #     plt.grid(True)
    #     plt.axis("equal")

    bidir_a_star_costmap = BidirectionalAStarCostmapPlanner(costmap, 4.0, obs)
    rx, ry = bidir_a_star_costmap.planning(sx, sy, gx, gy)
    plan = time.time()
    # bazier_smooth = BezierSmooth(rx, ry)
    # new_rx, new_ry, _ = bazier_smooth.smooth()
    new_rx, new_ry = rx, ry

    if show_animation:  # pragma: no cover
        plt.plot(rx, ry, "-r")
        plt.plot(new_rx, new_ry, "-b")
        plt.pause(.0001)
        plt.show()
    new_rx, new_ry = np.array(new_rx) / 25.0, np.array(new_ry) / 25.0

    return new_rx, new_ry


if __name__ == '__main__':
    
    #导入环境
    # unity_env = UnityEnvironment("1.x86_64")
    # env = UnityToGymWrapper(unity_env)

    env = CogEnvDecoder.CogEnvDecoder(env_name='1.x86_64')

    ap = Apriltag()
    ap.create_detector(debug=True)

    low_yellow = np.array([20, 30, 30])
    high_yellow = np.array([150, 255, 255])

    #创建标签与坐标点的对应关系
    position = {'A_x':-10,'A_y':-10,'B_x':-10,'B_y':-10,'C_x':-10,'C_y':-10,'D_x':-10,'D_y':-10,'E_x':-10,'E_y':-10}
        
    reward_sum = 0

    observation_init = []

    list_seq = []

    

    while(True): 

        a = random.randint(4,8)
        list_seq.append(a)
        b = 0
        c = 0
        d = 0
        e = 0
        while(True):
            b = random.randint(4,8)
            if(a != b):
                break
        list_seq.append(b)
        while(True):
            c = random.randint(4,8)
            if(a != c and b != c):
                break
        list_seq.append(c)
        while(True):
            d = random.randint(4,8)
            if(d != a and d != b and d != c):
                break
        list_seq.append(d)
        while(True):
            e = random.randint(4,8)
            if(e != a and e != b and e != c and e != d):
                break
        list_seq.append(e)
        # print(list_seq)
        for i in range(5):

            observation = env.reset()
            done = False

            sx = observation[1][0]
            sy = observation[1][1]
            gx = observation[list_seq[i]][0]
            gy = observation[list_seq[i]][1]
            observation_init.append(gx)
            observation_init.append(gy)
            flag_final = observation[list_seq[i]][2]
            path = main(sx, sy, gx, gy, observation)
            
            cx = path[0]
            cy = path[1]
            
            state = State(x=sx, y=sy, yaw=observation[1][2], v=0.0)
            T = 100.0  # max simulation time

            lastIndex = len(cx) - 1
            time1 = 0.0
            states = States()
            states.append(time1, state)
            target_course = TargetCourse(cx, cy)
            target_ind, _ = target_course.search_target_index(state)
            path = [[cx[i], cy[i]] for i in range(len(cx))]
            speed_p = TrapzoidPlanner()
            trajectory = speed_p.plan(path)
            trajectory[0] = trajectory[1]
            delta_dist = state.calc_distance(cx[-1], cy[-1])
            list = []
            
            while T >= time1 and not done:

                state.update_state(x=observation[1][0], y=observation[1][1], yaw=observation[1][2])
                di, target_ind = pure_pursuit_control(state, target_course, target_ind)
                target_speed = trajectory[target_ind][-1]
                ai = proportional_control(target_speed, state.v)

                control_vx, control_vy, control_vw = state.calc_control(ai, di, reverse=False)
                actions = [control_vx, control_vy, control_vw]

                observation, reward, done, info = env.step(actions)
                reward_sum += reward
                
                # img_length = int(observation[90016])
                # if img_length > 0:
                # float_img = np.asarray(observation[16:16+img_length], dtype=np.uint8)
                # b = bytearray()
                # for ele in float_img:
                #     b += ele.tobytes()
                # byteio = io.BytesIO(b)
                # img = Image.open(byteio)
                # img2 = img.convert('RGB')
                # opencv_img = np.array(img2)
                # frame = opencv_img[:,:,:].copy()

                frame = observation[0]
            


                low_yellow = np.array([20, 30, 30])
                high_yellow = np.array([150, 255, 255])

                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                mask_yellow = cv2.inRange(hsv, low_yellow, high_yellow)

                res = cv2.bitwise_and(frame,frame, mask= mask_yellow)
                res = res+255

                id = -1
                detections, id, contours = ap.detect(res)
                dis = -1
                for detection in detections:
                    point,rvec_cam,tvec_cam = tud.get_pose_point(detection.homography)

                    dis = tud.get_distance(detection.homography,25501)
                
                if id != -1 and delta_dist < 1.2 and dis != -1:
                    if id == 0 or id == 1:
                        position['A_x'] = gx
                        position['A_y'] = gy
                    elif id == 2 or id == 3:
                        position['B_x'] = gx
                        position['B_y'] = gy
                    elif id == 4:
                        position['C_x'] = gx
                        position['C_y'] = gy
                    elif id == 5:
                        position['D_x'] = gx
                        position['D_y'] = gy
                    elif id == 6 or id == 7:
                        position['E_x'] = gx
                        position['E_y'] = gy

                state.update_mec(ai, di)  # Control vehicle
                time1 += dt
                states.append(time1, state)
                state.update_state(x=observation[1][0], y=observation[1][1], yaw=observation[1][2])
                
                

                delta_dist = state.calc_distance(cx[-1], cy[-1])
                target_yaw = math.atan2(gy - observation[1][1], gx-observation[1][0])
                target_yaw = target_yaw * 57.3
                yaw = observation[1][2]*57.3
                delta_yaw = abs(target_yaw - yaw)
                if(delta_yaw >180):
                    delta_yaw = 360 - delta_yaw
                # if delta_dist<1 and delta_yaw<30:
                    # if (reward>0 and reward % 100 == 0):
                if delta_dist<0.5:
                    break
                # Test

            print('position:',position)
            assert lastIndex >= target_ind, "Cannot goal"
        print("observation_init:",observation_init)
        None_list = [] #保存所有空标签的位置
        for i in range(5):
            gx = observation_init[i*2]
            gy = observation_init[i*2+1]
            if position['A_x'] == gx:
                continue
            elif position['B_x'] == gx:
                continue
            elif position['C_x'] == gx:
                continue
            elif position['D_x'] == gx:
                continue
            elif position['E_x'] == gx:
                continue
            else:
                None_list.append(gx)
                None_list.append(gy)

        if len(None_list)<=4:
            break
        
    observation = env.reset()
     
    op = 0
    for temp in range(4, 9, 1):
        if(observation[temp][2]==True):
            op += 1
    
    print("op:",op)
    
    while not observation[4][2] or not observation[5][2] or not observation[6][2] or not observation[7][2] or not observation[8][2]:

        observation = env.reset()
        
        sx = observation[1][0]
        sy = observation[1][1]

        if op > 4:
            op = 0

        if op == 0:
            gx = position['A_x']
            gy = position['A_y']
        elif op == 1:
            gx = position['B_x']
            gy = position['B_y']
        elif op == 2:
            gx = position['C_x']
            gy = position['C_y']
        elif op == 3:
            gx = position['D_x']
            gy = position['D_y']
        elif op == 4:
            gx = position['E_x']
            gy = position['E_y']
        
        if gx == -10:
            for j in range(0,len(None_list),2):
                sx = observation[1][0]
                sy = observation[1][1]
                gx = None_list[j]
                gy = None_list[j+1]
                path = main(sx, sy, gx, gy, observation)
    
                cx = path[0]
                cy = path[1]

                state = State(x=sx, y=sy, yaw=observation[1][2], v=0.0)

                lastIndex = len(cx) - 1
                states = States()
                states.append(time1, state)
                target_course = TargetCourse(cx, cy)
                target_ind, _ = target_course.search_target_index(state)
                path = [[cx[i], cy[i]] for i in range(len(cx))]
                speed_p = TrapzoidPlanner()
                trajectory = speed_p.plan(path)
                trajectory[0] = trajectory[1]
                delta_dist = state.calc_distance(cx[-1], cy[-1])
                flag = False
                while not done:

                    state.update_state(x=observation[1][0], y=observation[1][1], yaw=observation[1][2])
                    di, target_ind = pure_pursuit_control(state, target_course, target_ind)
                    target_speed = trajectory[target_ind][-1]
                    ai = proportional_control(target_speed, state.v)

                    control_vx, control_vy, control_vw = state.calc_control(ai, di, reverse=False)
                    actions = [control_vx, control_vy, control_vw]

                    observation, reward, done, info = env.step(actions)
                    if reward > 0:
                        flag = True
                        break
                    state.update_mec(ai, di)  # Control vehicle
                    states.append(time1, state)
                    state.update_state(x=observation[1][0], y=observation[1][1], yaw=observation[1][2])
                    delta_dist = state.calc_distance(cx[-1], cy[-1])
                    target_yaw = math.atan2(gy - observation[1][1], gx-observation[1][0])
                    target_yaw = target_yaw * 57.3
                    yaw = observation[1][2]*57.3
                    delta_yaw = abs(target_yaw - yaw)
                    if(delta_yaw >180):
                        delta_yaw = 360 - delta_yaw
                    if delta_dist< 0.5 or reward > 0:
                        break
                if flag == True:
                    break    

        else: 
            path = main(sx, sy, gx, gy, observation)
    
            cx = path[0]
            cy = path[1]

            state = State(x=sx, y=sy, yaw=observation[1][2], v=0.0)

            lastIndex = len(cx) - 1
            states = States()
            states.append(time1, state)
            target_course = TargetCourse(cx, cy)
            target_ind, _ = target_course.search_target_index(state)
            path = [[cx[i], cy[i]] for i in range(len(cx))]
            speed_p = TrapzoidPlanner()
            trajectory = speed_p.plan(path)
        
            trajectory[0] = trajectory[1]
            delta_dist = state.calc_distance(cx[-1], cy[-1])
            while not done:

                state.update_state(x=observation[1][0], y=observation[1][1], yaw=observation[1][2])
                di, target_ind = pure_pursuit_control(state, target_course, target_ind)
                target_speed = trajectory[target_ind][-1]
                ai = proportional_control(target_speed, state.v)

                control_vx, control_vy, control_vw = state.calc_control(ai, di, reverse=False)
                actions = [control_vx, control_vy, control_vw]

                observation, reward, done, info = env.step(actions)
                state.update_state(x=observation[1][0], y=observation[1][1], yaw=observation[1][2])
                delta_dist = state.calc_distance(cx[-1], cy[-1])
                if reward>0 or delta_dist < 0.5:
                    break
                state.update_mec(ai, di)  # Control vehicle
                states.append(time1, state)
        op += 1
    
    j = 0
    while j<100:
        actions = [0.1, 0.1, 0.1] 
        observation, reward, done, info = env.step(actions)
            
        # if show_animation:  # pragma: no cover
        #     plt.cla()
        #     plt.plot(cx, cy, ".r", label="course")
        #     plt.plot(states.x, states.y, "-b", label="trajectory")
        #     plt.legend()
        #     plt.xlabel("x[m]")
        #     plt.ylabel("y[m]")
        #     plt.axis("equal")
        #     plt.grid(True)

        #     plt.subplots(1)
        #     plt.plot(states.t, [iv * 3.6 for iv in states.v], "-r")
        #     plt.xlabel("Time[s]")
        #     plt.ylabel("Speed[km/h]")
        #     plt.grid(True)
        #     plt.show()

        
            