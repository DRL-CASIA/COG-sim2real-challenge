#pip install CogEnvDecoder==1.0.0
from Cogenvdecoder.CogEnvDecoder import CogEnvDecoder
import numpy as np
import cv2
import time
def check_state(state, info=None):
    image_data = state["color_image"]
    laser_data = np.array(state["laser"])
    vector_data = state["vector"]
    print("=======================state check====================")
    print("image shape: {}".format(image_data.shape))
    # laser scan distances from -135 deg to +135 deg, scan angle resolution is 270/(61-1) 
    print("laser shape: {}, max distance: {}, min distance: {}".format(laser_data.shape, np.max(laser_data), np.min(laser_data)))
    # self_pose: [x, y, theta(rad)], self_info: [remaining HP, remaining bullet]
    # enemy_pose: [x, y, theta(rad)], enemy_info: [remaining HP, remaining bullet]
    print("self pose: {}, self info: {}, enemy active: {}, enemy pose: {}, enemy_info: {}".format(vector_data[0], vector_data[1], vector_data[2], vector_data[3], vector_data[4]))
    # self_velocity: [vx, vy, vw]
    print("self velocity: {}".format(vector_data[11]))
    # goal_x: [x, y, is_activated?]
    print("goal 1: {}, goal 2: {}, goal 3: {}, goal 4: {}, goal 5:{}".format(vector_data[5], vector_data[6], vector_data[7], vector_data[8], vector_data[9]))
    # total counts of collisions, total collision time
    print("total collisions: {}, total collision time: {} ".format(vector_data[10][0], vector_data[10][1]))
    if info is not None:
        print("Number of goals have been activated: {}".format(info[1][3]))
        # attack damage is blue one caused damage to red one
        print("time taken: {}, attack damage: {}, score: {}".format(info[1][1], info[1][2], info[1][0]))
    print("-----------------------end check---------------------")


env = CogEnvDecoder(env_name="stage2/reality_linux_v3.0/cog_sim2real_env.x86_64", no_graphics=False, 
                    time_scale=1, worker_id=2, seed=1234, force_sync=True) # linux os
# env = CogEnvDecoder(env_name="win_V1/RealGame.exe", no_graphics=False, time_scale=1, worker_id=1) # windows os
# env_name: path of the simulator
# no_graphics: should use headless mode [Warning: if no_graphics is True, image if invalid!]
# time_scale: useful for speedup collecting data during training, max value is 100
# worker_id: socket port offset, useful for multi-thread training
# seed: random seed for generate position of the goals and the robots
# force_sync: use time synchronization (True) or not (False)
num_episodes = 10
num_steps_per_episode = 500 # max: 1500
for i in range(num_episodes):
    #every time call the env.reset() will reset the envinronment
    observation = env.reset(fri_cor=0.1, KP=8, KI=0, KD=2, VK1=0.375, M=3.4, Wheel_I=0.0125)
    # fir_cor: friction factor
    # KP, KI, KD: parameters of PID
    # VK1: parameter of the motor(M3508I)
    # M: the mass of the robot
    # Wheel_I: the inertia of the wheel
    
    for j in range(num_steps_per_episode):
        action = env.action_space.sample()
        # action = [0.5, 0.5, 0.1, 0]  # [vx, vy, vw, fire]; vx: the velocity at which the vehicle moves forward, vy: the velocity at which the vehicle moves to the left, vw: Angular speed of the vehicle counterclockwise rotation, fire: Shoot or not
        obs, reward, done, info = env.step(action)
        cv2.imshow("color_image", obs["color_image"])
        cv2.waitKey(1)
        check_state(obs, info)
        if done:
            break
        
