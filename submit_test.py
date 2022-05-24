from Cogenvdecoder.CogEnvDecoder import CogEnvDecoder
import numpy as np
from cog_agent import Agent

env = CogEnvDecoder(env_name="mac_confrontation_v2/cog_confrontation_env.app", no_graphics=False, time_scale=1, worker_id=1) 

num_eval_episodes = 10

eval_agent = Agent(model_path="")

activated_goals_analy = []
time_token_analy = []
attack_damage_analy = []
score_analy = []
for i in range(num_eval_episodes):

    obs = env.reset()
    done = False
    info = None
    while not done:
        action = eval_agent.agent_control(obs=obs, done=done, info=info)
        obs, reward, done, info = env.step(action)

    num_activted_goals = info[1][3]
    activated_goals_analy.append(num_activted_goals)
    time_token = info[1][1]
    time_token_analy.append(time_token)
    attack_damage = info[1][2]
    attack_damage_analy.append(attack_damage)
    score = info[1][0]
    score_analy.append(score)

mean_activated_goal = np.mean(activated_goals_analy)
mean_time_token = np.mean(time_token_analy)
mean_attack_damage = np.mean(attack_damage_analy)
mean_score = np.mean(score_analy)
print("mean activated goal: {}, mean time token: {}, mean attack damage: {}, mean score: {}".format(
    mean_activated_goal, mean_time_token, mean_attack_damage, mean_score))
