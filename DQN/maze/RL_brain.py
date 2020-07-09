import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from maze_env import Maze


BATCH_SIZE = 32
LR = 0.01    # learning rate
EPSILON = 0.9  # 最优选择动作百分比
GAMMA = 0.9  # 奖励递减参数
TARGET_REPLACE_ITER = 100  # Q 现实网络的更新频率
MEMORY_CAPACITY = 2000  # 记忆库大小
env = Maze()
N_ACTIONS = env.n_actions
N_STATES = env.n_features
# ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(
# ), int) else env.action_space.sample().shape  # to confirm the shape


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 10)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization
        self.out = nn.Linear(10, N_ACTIONS)
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
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))  # 初始化记忆库 每一行存储两个state，一个reward,一个action
        self.optimizer = torch.optim.Adam(
            self.eval_net.parameters(), lr=LR)  # torch的优化器
        self.loss_func = nn.MSELoss()  # 误差公式

    def choose_action(self, x):
        """
        输入：状态x
        输出：动作action
        """
        x = torch.unsqueeze(torch.FloatTensor(x), 0) #维度扩充
        if np.random.uniform() < EPSILON:
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            # action = action[0] if ENV_A_SHAPE == 0 else action.reshape(
            #     ENV_A_SHAPE)  # return the argmax index
        else:  #随机选取动作
            action = np.random.randint(0, N_ACTIONS)
            # action = action if ENV_A_SHAPE == 0 else action.reshape(
            #     ENV_A_SHAPE)
        return action

    def store_transition(self, s, a, r, s_):
        #存储记忆
        transition = np.hstack((s, [a, r], s_))
        #如果记忆库满了，就覆盖老数据
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        #target网络更新
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        #学习记忆库中的记忆
        #抽取记忆库中的批数据
        sample_index = np.random.choice(
            MEMORY_CAPACITY, BATCH_SIZE)  # 相当于从0到MEMORY_CAPACITY取BATCH_SIZE个数
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES + 1: N_STATES + 2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        #q_eval w.r.t the action in experience
        # shape (batch,1)    取对角线元素并以列的形式（1维）输出
        q_eval = self.eval_net(b_s).gather(1, b_a)
        # 输入b_s_以得到下一步的预测动作。detach保持一部分的网络参数不变，切断反向传播
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + GAMMA * \
            q_next.max(1)[0].view(BATCH_SIZE, 1)  # 将batch个结果拼接成一行
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
