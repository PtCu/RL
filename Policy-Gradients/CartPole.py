import torch
import gym
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import math
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from itertools import count
env = gym.make('CartPole-v1')


class PG_network(nn.Module):
    def __init__(self):
        super(PG_network, self).__init__()
        self.linear1 = nn.Linear(4, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.linear2 = nn.Linear(128, 2)

        # self
        # self.optimizer = optim.Adam(self.parameters(),lr=1e-2)

    def forward(self, x):
        x = self.linear1(x)
        x = self.dropout(x) #减少过拟合
        x = F.relu(x)
        action_scores = self.linear2(x)
        # x = self.dropout(x)
        # x = F.relu(x).unsqueeze(0)
        # x = x.unsqueeze(0)
        return F.softmax(action_scores, dim=1)
        # maxvalue,index = torch.max(x,dim=1)
        # y = x.squeeze(0)
        # action_random = np.random.choice(y.detach().numpy())
        # print(action_random)
        # return x


policyG_object = PG_network()
optimizer = optim.Adam(policyG_object.parameters(), lr=1e-2)
possibility_store = []
r_store = []


def choose_action(s):
    s = torch.from_numpy(s).float().unsqueeze(0)    #state
    probs = policyG_object(s)
    """
    作用是创建以参数probs为标准的类别分布，样本是来自 “0 … K-1” 的整数，其中 K 是probs参数的长度。
    也就是说，按照传入的probs中给定的概率，在相应的位置处进行取样，取样返回的是该位置的整数索引。

    如果 probs 是长度为 K 的一维列表，则每个元素是对该索引处的类进行抽样的相对概率。

    如果 probs 是二维的，它被视为一批概率向量。

    """
    m = Categorical(probs)
    action = m.sample()
    b = m.log_prob(action)

    possibility_store.append(m.log_prob(action))
    return action.item()


alpha = 0.9
gammar = 0.9
reward_delay = 0.9
# finfo函数是根据height.dtype类型来获得信息，获得符合这个类型的float型，eps是取非负的最小值。
eps = np.finfo(np.float64).eps.item()
# R_store = []


def policy_gradient_learn():
    R = 0
    R_store = []
    delta_store = []
    # theta = -torch.log10()
    for r in r_store[::-1]:
        R = r + reward_delay*R
        R_store.insert(0, R)
    R_store = torch.tensor(R_store)
    R_store = (R_store - R_store.mean())/(R_store.std()+eps)

    # zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的对象，这样做的好处是节约了不少的内存。
    for p, v in zip(possibility_store, R_store):
        delta_store.append(-p*v)
    optimizer.zero_grad()

    delta_store = torch.cat(delta_store).sum()  #cat:拼接两个序列

    delta_store.backward()
    optimizer.step()
    del possibility_store[:]  # del删除的是变量，而不是数据。
    del r_store[:]
    # print(loss)


def main():
    running_reward = 10
    for i_episode in count(1):
        s, ep_reward = env.reset(), 0
        for t in range(1, 10000):
            # env.render()
            a = choose_action(s)
            s, r, done, info = env.step(a)
            r_store.append(r)
            ep_reward += r
            # print(r,a)
            if done:
                break

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        policy_gradient_learn()
        if i_episode % 10 == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                i_episode, ep_reward, running_reward))
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))

        # torch.save(policy.state_dict(),'hello.pt')
if __name__ == '__main__':
    main()
