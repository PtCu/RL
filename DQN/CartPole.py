import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
BATCH_SIZE = 32
LR = 0.01    # learning rate
EPSILON = 0.9  # 最优选择动作百分比
GAMMA = 0.9  # 奖励递减参数
TARGET_REPLACE_ITER = 100  # Q 现实网络的更新频率
MEMORY_CAPACITY = 2000  # 记忆库大小
env = gym.make('CartPole-v0')  # 立杆子游戏
env = env.unwrapped
N_ACTOPNS = env.action_space.n  # 杆子能做的动作
N_STATES = env.observation_space.shape[0]  # 杆子能获取的环境信息数
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(
), int) else env.action_space.sample().shape  # to confirm the shape


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 10)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization
        self.out = nn.Linear(10, N_ACTOPNS)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


class DQN(object):
    def __init__(self):
        # 建立 target net 和 eval net 还有 memory
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0  # 用于target更新计时
        self.memory_counter = 0  # 记忆库记数
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))  # 初始化记忆库
        self.optimizer = torch.optim.Adam(
            self.eval_net.parameters(), lr=LR)  # torch的优化器
        self.loss_func = nn.MSELoss()  # 误差公式

    def choose_action(self, x):  # 用eval来选动作
        """
        输入：状态x
        输出：动作action
        """
        x = torch.unsqueeze(torch.FloatTensor(x), 0)  # 维度扩充
        if np.random.uniform() < EPSILON:
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(
                ENV_A_SHAPE)  # return the argmax index
        else:  # 随机选取动作
            action = np.random.randint(0, N_ACTOPNS)
            action = action if ENV_A_SHAPE == 0 else action.reshape(
                ENV_A_SHAPE)
        return action

    def store_transition(self, s, a, r, s_):
        # 存储记忆
        transition = np.hstack((s, [a, r], s_))
        # 如果记忆库满了，就覆盖老数据
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    """
    type()	返回数据结构类型（list、dict、numpy.ndarray 等）
    dtype()	返回数据元素的数据类型（int、float等）
    astype()改变np.array中所有数据元素的数据类型。备注：能用dtype() 才能用 astype()
    """

    def learn(self):
        # target网络更新
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # 学习记忆库中的记忆
        # 抽取记忆库中的批数据
        sample_index = np.random.choice(
            MEMORY_CAPACITY, BATCH_SIZE)  # 相当于从0到MEMORY_CAPACITY取BATCH_SIZE个数
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES + 1: N_STATES + 2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        # q_eval w.r.t the action in experience
        # q_估计
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch,1)
        # target_net用的是很久之前的参数，用来预测q现实
        # 输入b_s_以得到下一步的预测动作。detach保持一部分的网络参数不变，切断反向传播
        q_next = self.target_net(b_s_).detach()
        # q_现实
        q_target = b_r + GAMMA * \
            q_next.max(1)[0].view(BATCH_SIZE, 1)  # 将batch个结果拼接成一行
        loss = self.loss_func(q_eval, q_target)  # 反向传播更新eval_net

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


dqn = DQN()

print('\nCollecting experience...')
for i_episode in range(400):
    s = env.reset()
    ep_r = 0
    while True:
        # 刷新环境
        env.render()

        # DQN根据观测值选择行为
        a = dqn.choose_action(s)

        # 环境根据行为给出下一个state,reward,是否终止
        s_, r, done, info = env.step(a)

        # modify the reward
        x, x_dot, theta, theta_dot = s_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / \
            env.theta_threshold_radians - 0.5
        r = r1 + r2

        # DQN存储记忆
        dqn.store_transition(s, a, r, s_)

        ep_r += r
        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()
            if done:
                print('Ep: ', i_episode,
                      '| Ep_r: ', round(ep_r, 2))

        if done:
            break

        # 将下一个state_变为下次循环的state
        s = s_
