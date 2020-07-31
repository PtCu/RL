import gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available()
                      else "cpu")  # if gpu is to be used


class FrameProcessor():
    def __init__(self, im_size=84):
        self.im_size = im_size

    def process(self, frame):
        im_size = self.im_size
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = frame[46:160+46, :]

        frame = cv2.resize(frame, (im_size, im_size),
                           interpolation=cv2.INTER_LINEAR)
        frame = frame.reshape((1, im_size, im_size))

        x = torch.from_numpy(frame)
        return x


def fp(n_frame):
    n_frame = torch.from_numpy(n_frame)
    h = n_frame.shape[-2]
    return n_frame.view(1, h, h)


c, h, w = fp(env.reset()).shape


class CNet(nn.Module):
    def __init__(self, outputs, device): #outputs=n_actions
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, bias=False)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=False)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False)
        self.fc1 = nn.Linear(64*7*7, 512)
        self.fc2 = nn.Linear(512, outputs)
        self.device = device

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            m.bias.data.fill_(0.0)
        if type(m) == nn.Conv2d:
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def forward(self, x):
        x = x.to(self.device).float() / 255.
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        return self.fc2(x)


class DQN(object):
    def __init__(self,  batch_size, n_actions, GAMMA, INITIAL_EPSILON, FINAL_EPSILON,  EPS_DECAY, TARGET_UPDATE, eval_net, target_net, replay_memory, lr=6.25e-5, eps=1.5e-4):

        self.lr = lr
        self.eps = eps

        self.memory = replay_memory  # 初始化记忆库

        # 用于做决策的参数
        self._eps = INITIAL_EPSILON
        self._FINAL_EPSILON = FINAL_EPSILON
        self._INITIAL_EPSILON = INITIAL_EPSILON
        self._EPS_DECAY = EPS_DECAY
        self._n_actions = n_actions
        self._device = device

        # 神经网络  估计和目标
        """
        q_target:学会总结. 打乱相关性
        反向传播真正训练的网络是只有一个，就是eval_net。
        target_net 只做正向传播得到q_target (q_target = r +γ*max Q(s,a)). 
        其中 Q(s,a)是若干个经过target-net正向传播的结果。
        """
        self.eval_net = eval_net
        self.target_net = target_net

        self.optimizer = torch.optim.Adam(
            self.eval_net.parameters(), lr, eps)  # torch的优化器
        self.loss_func = nn.MSELoss()  # 误差公式

        self.TARGET_UPDATE = TARGET_UPDATE
   
        self.learn_step_counter = 0

    def choose_action(self, state, training=False):  # 用eval来选动作
        """
        输入：状态state
        输出：动作action
        """
        sample = random.random()
        if training:
            self._eps -= (self._INITIAL_EPSILON -
                          self._FINAL_EPSILON)/self._EPS_DECAY
            self._eps = max(self._eps, self._FINAL_EPSILON)
        if sample > self._eps:
            with torch.no_grad():
                a = self.eval_net(state).max(1)[1].cpu().view(1, 1)
        else:
            a = torch.tensor([[random.randrange(self._n_actions)]],
                             device='cpu', dtype=torch.long)

        return a.numpy()[0, 0].item()

    def learn(self):
        #若干步后更新target
        if self.learn_step_counter % self.TARGET_UPDATE == 0:
            self.update_target()
        self.learn_step_counter += 1
        
        #从记忆库抽取记忆
        state_batch, action_batch, reward_batch, n_state_batch, done_batch = self.memory.sample()

        #估计Q值
        """
        gather: dim=1时，按列取数
        action_batch指明了所做过的动作的index，如[[4],[0],[4],[1],[1]] 表明记忆中这一batch做过的动作为4，0，4，1，1
        eval_net(state_batch)返回的是全部动作的价值，如
            [[ 0.4572,  0.3634,  0.2419, -0.1153, -0.5035,  0.0569],
            [ 0.4921,  0.3209,  0.0316, -0.0606, -0.2619,  0.1681],
            [ 0.3856,  0.2895,  0.1565, -0.0291, -0.3200,  0.1980],
            [ 0.4987,  0.3081,  0.0362, -0.0518, -0.2651,  0.1753],
            [ 0.3840,  0.2452,  0.0488, -0.0934, -0.3971,  0.0686],]
        q_eval要取做过的动作的价值，即
            [[-0.5035],
            [ 0.4921],
            [-0.3200],
            [ 0.3081],
            [ 0.2452],]
        """
        q_eval = self.eval_net(state_batch).gather(1, action_batch)

        #未来的Q值，target. 输入next_state，输出每个动作的价值。max(1)表示返回每一行的最大值。dim 0表示列，1表示行
        q_next = self.target_net(n_state_batch).max(1)[0].detach()
          
        #目标Q值。如果done的话就只加reward
        expected_state_action_values = (
            q_next * GAMMA) * (1. - done_batch[:, 0]) + reward_batch[:, 0]
            
        # Compute Huber loss
        loss = F.smooth_l1_loss(q_eval, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.eval_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def update_target(self):
        self.target_net.load_state_dict(eval_net.state_dict())


class ReplayMemory(object):
    def __init__(self, capacity, state_shape, n_actions, device, batch_size):
        c, h, w = state_shape
        self.capacity = capacity
        self.device = device
        self.m_states = torch.zeros((capacity, c, h, w), dtype=torch.uint8)
        self.m_actions = torch.zeros((capacity, 1), dtype=torch.long)
        self.m_rewards = torch.zeros((capacity, 1), dtype=torch.int8)
        self.m_dones = torch.zeros((capacity, 1), dtype=torch.bool)
        self.position = 0
        self.size = 0
        self.batch_size = batch_size

    def store_transition(self, state, action, reward, done):
        """Saves a transition."""
        self.m_states[self.position] = state  # 5,84,84
        self.m_actions[self.position, 0] = action
        self.m_rewards[self.position, 0] = reward
        self.m_dones[self.position, 0] = done
        self.position = (self.position + 1) % self.capacity
        self.size = max(self.size, self.position)

    def sample(self):
        i = torch.randint(0, high=self.size, size=(self.batch_size,))
        bs = self.m_states[i, :4]
        bns = self.m_states[i, 1:]
        ba = self.m_actions[i].to(self.device)
        br = self.m_rewards[i].to(self.device).float()
        bd = self.m_dones[i].to(self.device).float()
        return bs, ba, br, bns, bd

    def __len__(self):
        return self.size
