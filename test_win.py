#pip install CogEnvDecoder==0.1.0

from Cogenvdecoder import CogEnvDecoder


env = CogEnvDecoder.CogEnvDecoder(env_name='RealGame.exe')

for i in range(100):
    #每次调用env.reset()将重置环境，任务没有完成之前，请不要调用
    observation = env.reset()
    
    img = observation[0]                           #摄像头所获取到的图像信息

    sx = observation[1][0]                         #主车x方向位置、单位：m
    sy = observation[1][1]                         #主车y方向位置、单位：m
    syaw = observation[1][2]                       #主车在世界坐标系下的角度、单位：弧度
    
    

    esx = observation[2][0]                        #敌车x方向位置、单位：m
    esy = observation[2][1]                        #敌车y方向位置、单位：m
    eyaw = observation[2][2]                       #敌车在世界坐标系下的角度、单位：弧度
    


    HP = observation[3][0]                         #主车的血量
    Bullet = observation[3][1]                     #主车的弹药

    #五个目标点的位置随机给出
    goal_pos1 = observation[4]                     #x:[4][0] y:[4][1] 是否被激活：[4][2]
    goal_pos2 = observation[5]                     #x:[5][0] y:[5][1] 是否被激活：[5][2]
    goal_pos3 = observation[6]                     #x:[6][0] y:[6][1] 是否被激活：[6][2]
    goal_pos4 = observation[7]                     #x:[7][0] y:[7][1] 是否被激活：[7][2]
    goal_pos5 = observation[8]                     #x:[8][0] y:[8][1] 是否被激活：[8][2]

    collision_times = observation[9][0]            #碰撞次数
    collision_time = observation[9][1]             #碰撞时间
    
    
    j = 0
    while j<100:
        actions = [0.5, 0.5, 0.1, 0] 
        observation, reward, done, info = env.step(actions)  #算法输出动作与环境交互，得到新的观测量

        img = observation[0]                           #摄像头所获取到的图像信息
        
        sx = observation[1][0]                         #主车x方向位置、单位：m
        sy = observation[1][1]                         #主车y方向位置、单位：m
        syaw = observation[1][2]                       #主车在世界坐标系下的角度、单位：弧度
       

        esx = observation[2][0]                        #敌车x方向位置、单位：m
        esy = observation[2][1]                        #敌车y方向位置、单位：m
        eyaw = observation[2][2]                       #敌车在世界坐标系下的角度、单位：弧度

        HP = observation[3][0]                         #主车的血量
        Bullet = observation[3][1]                     #主车的弹药

        #五个目标点的位置随机给出
        goal_pos1 = observation[4]                     #x:[4][0] y:[4][1] 是否被激活：[4][2]
        goal_pos2 = observation[5]                     #x:[5][0] y:[5][1] 是否被激活：[5][2]
        goal_pos3 = observation[6]                     #x:[6][0] y:[6][1] 是否被激活：[6][2]
        goal_pos4 = observation[7]                     #x:[7][0] y:[7][1] 是否被激活：[7][2]
        goal_pos5 = observation[8]                     #x:[8][0] y:[8][1] 是否被激活：[8][2]

        collision_times = observation[9][0]            #碰撞次数
        collision_time = observation[9][1]             #碰撞时间

        j += 1