import torch
import gym
from RL_brain import Policy
from torch.distributions import Categorical

model = Policy()
model.load_state_dict(torch.load('hello.pt'))
model.eval()


def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = model(state)
    m = Categorical(probs)
    action = m.sample()
    return action.item()


env = gym.make('CartPole-v1')
t_all = []
for i_episode in range(2):
    observation = env.reset()
    for t in range(10000):
        env.render()
        cp, cv, pa, pv = observation
        action = select_action(observation)
        observation, reward, done, info = env.step(action)
        if done:
            # print("Episode finished after {} timesteps".format(t+1))
            print("倒了")
            t_all.append(t)
            break
env.close()
print(t_all)
print(sum(t_all)/len(t_all))
