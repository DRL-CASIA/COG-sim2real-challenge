from Cogenvdecoder import CogEnvDecoder

env = CogEnvDecoder.CogEnvDecoder(env_name='1.x86_64')

for i in range(100):
    observation = env.reset()
    img = observation[0] #摄像头信息
    sx = observation[1][0] #
    sy = observation[1][1] #主车位置

    esx = observation[2][0] #
    esy = observation[2][1] #敌车位置

    HAB = observation[3]  #主车的血量和弹药

    goal_pos1 = observation[4] #x:[4][0] y:[4][1]
    goal_pos2 = observation[5] #x:[5][0] y:[5][1]
    goal_pos3 = observation[6] #x:[6][0] y:[6][1]
    goal_pos4 = observation[7] #x:[7][0] y:[7][1]
    goal_pos5 = observation[8] #x:[8][0] y:[8][1]

    j = 0
    while j<10000:
        actions = [0.5, 0.5, 0.1] 
        observation, reward, done, info,judge_result = env.step(actions)