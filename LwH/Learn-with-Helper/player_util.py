from __future__ import division
import math
import numpy as np
import torch
from torch.autograd import Variable
from utils import normal
from policy_domain import Policy_Domain


class Agent(object):
    def __init__(self, model, env, args, state, rank):
        self.time_step = 0
        self.model = model
        self.env = env
        self.state = state
        self.eps_len = 0
        self.args = args
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.infos = []
        self.entropies = []
        self.done = True
        self.reward = 0
        self.info = None
        self.rank = rank
        self.action_pre = []
        self.action_pre_sup = []
        self.reset_flag = False
        self.action_test_collection = []

        self.prior = Policy_Domain(env.observation_space, env.action_space)

    def action_train(self):  # 
        self.time_step += 1
        # model为A3C，此步为前向计算forward。value network——输入state，输出value;policy network——输入state,输出mean和标准差(theta)
        value, mu_learned, sigma_learned = self.model(Variable(self.state))

        if self.args.use_prior:
            mu_prior, sigma_prior = self.prior.forward(Variable(
                self.state), self.time_step, self.args)  # prior network 输入state，输出mean和标准差(h)
            sigma_prior = sigma_prior.diag()

        sigma_learned = sigma_learned.diag()

        self.reset_flag = False

        if self.args.use_prior:  # sigma_prior对应论文中的sigma_h,   sigma_learned对应论文中的sigma_theta
            sigma = (sigma_learned.inverse() + sigma_prior.inverse()
                     ).inverse()  # 计算behavior的方差 公式(23)
            temp = torch.matmul(sigma_learned.inverse(), mu_learned) + \
                torch.matmul(sigma_prior.inverse(), mu_prior)
            mu = torch.matmul(sigma, temp)  # mean_behavior 公式(24)
        else:
            sigma = sigma_learned
            mu = mu_learned

        sigma = sigma.diag()  # sigma_behavior(behavior policy)
        sigma_learned = sigma_learned.diag()

        eps = torch.randn(mu.size())    #随机数 randn生成正态分布的随机数
        pi = np.array([math.pi])
        pi = torch.from_numpy(pi).float()
        eps = Variable(eps)
        pi = Variable(pi)
        #?
        action = (mu + sigma.sqrt() * eps).data  # 根据均值加方差确定动作

        act = Variable(action)
        prob = normal(act, mu, sigma)  # 计算正态分布（22）behavior policy
        # execute the action
        action = torch.clamp(
            action, self.env.action_space.low[0], self.env.action_space.high[0])
        # expand_as():把一个tensor变成和函数括号内一样形状的tensor
        entropy = 0.5 * \
            ((sigma_learned * 2 * pi.expand_as(sigma_learned)).log() + 1)
        self.entropies.append(entropy)
        log_prob = (prob + 1e-6).log()
        self.log_probs.append(log_prob)
        state, reward, self.done, self.info = self.env.step(
            action.cpu().numpy())

        # self.env.render()
        self.state = torch.from_numpy(state).float()
        self.eps_len += 1
        self.done = self.done

        self.values.append(value)
        self.rewards.append(reward)
        self.infos.append(self.info)
        return self

    def action_test(self):
        with torch.no_grad():
            value, mu, sigma = self.model(Variable(self.state))

        action = mu.data
        # # eps = torch.randn(mu.size())
        # # action = (mu + sigma.sqrt() * eps).data  # 根据均值加方差确定动作
        # # execute the action
        action = torch.clamp(
            action, self.env.action_space.low[0], self.env.action_space.high[0])
        self.action_test_collection.append(action.cpu().numpy())
        state, self.reward, self.done, self.info = self.env.step(
            action.cpu().numpy())

        self.state = torch.from_numpy(state).float()
        self.eps_len += 1
        self.done = self.done or self.eps_len >= self.args.max_episode_length
        return self

    def clear_actions(self):
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.action_pre = []
        self.action_pre_sup = []
        return self
