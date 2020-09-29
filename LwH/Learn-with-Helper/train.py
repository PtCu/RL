from __future__ import division
from setproctitle import setproctitle as ptitle
from utils import setup_logger
import logging
import numpy as np
import torch
import torch.optim as optim
from environment import create_env
from utils import ensure_shared_grads
from model import A3C_MLP
from player_util import Agent
from torch.autograd import Variable
import os
from torch.utils.tensorboard import SummaryWriter


def train(rank, args, shared_model, optimizer):  # optimizer为shared_model的
    init = True
    ptitle('Training Agent: {}'.format(rank))
    torch.manual_seed(args.seed + rank)
    env = create_env(args.env, args.seed + rank)
    # env = gym.make(args.env)
    # env.seed(args.seed + rank)

    if optimizer is None:
        if args.optimizer == 'RMSprop':
            optimizer = optim.RMSprop(shared_model.parameters(), lr=args.lr)
        if args.optimizer == 'Adam':
            optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)

    player = Agent(None, env, args, None, rank)
    player.model = A3C_MLP(
        player.env.observation_space, player.env.action_space, args.stack_frames)
    player.state = player.env.reset()
    player.state = torch.from_numpy(player.state).float()
    player.model.train() #固定使用场景为train

    if rank == 1:
        # file = open(os.path.join(args.log_dir, 'TD_Error.txt'), 'w+')
        writer=SummaryWriter('8_27_train')

    local_step_counter = 0
    while True:
        if init:  # 初始化
            shared_model.training_steps.weight.data \
                .copy_(torch.Tensor([0]))
            shared_model.training_steps.bias.data \
                .copy_(torch.Tensor([0]))
            init = False
        player.model.load_state_dict(
            shared_model.state_dict())  # synchronize parameters
        for step in range(args.num_steps):
            # print("thread", rank, local_step_counter, shared_model.training_steps.weight.data.cpu().numpy())
            local_step_counter += 1  # update step counters
            shared_model.training_steps.weight.data \
                .copy_(torch.Tensor([1]) + shared_model.training_steps.weight.data)  # 总步骤（各个worker所走步数之和）T每次加一

            player.action_train()  # core
            if player.done:
                break

        terminal = False
        if player.done or player.eps_len >= args.max_episode_length:  # 玩家完成或者超出最大迭代次数
            terminal = True
            shared_model.done_nums+=1
            if 'is_success' in player.info.keys():
                shared_model.success_num+=1

        R = torch.zeros(1)
        if not player.done:  # 结算
            state = player.state
            # A3C，value和policy net是用的同一个网络
            value, _, _ = player.model(Variable(state))
            R = value.data

        if terminal:    #重置
            shared_model.training_steps.bias.data \
                .copy_(torch.Tensor([1]) + shared_model.training_steps.bias.data)  # 总步数加一
            player.eps_len = 0
            state = player.env.reset()
            player.state = torch.from_numpy(state).float()
            player.reset_flag = True

        player.values.append(Variable(R))
        policy_loss = 0
        value_loss = 0
        R = Variable(R)
        gae = torch.zeros(1, 1)
        for i in reversed(range(len(player.rewards))):
            R = args.gamma * R + np.float(player.rewards[i])  # reward
            advantage = R - player.values[i]  # advantage
            value_loss = value_loss + 0.5 * advantage.pow(2)  # 公式(10) 更新w
            if rank == 1:
                # file.write(str(advantage.pow(2).data.cpu().numpy()[0]))
                # file.write(' ')
                # file.write(
                #     str(int(shared_model.training_steps.weight.data.cpu().numpy()[0])))
                # file.write('\n')
                writer.add_scalar('TD-error/train', advantage.pow(2).data.cpu().numpy()[0],
                                  shared_model.training_steps.weight.data.cpu().numpy()[0])

            player.values[i] = player.values[i].float()
            player.values[i+1] = player.values[i+1].float()
            delta_t = player.rewards[i] + args.gamma * \
                player.values[i + 1].data - \
                player.values[i].data  # a2c计算td-error
            # GAE算法
            gae = gae * args.gamma * args.tau + delta_t

            policy_loss = policy_loss - \
                (player.log_probs[i].sum() * Variable(gae)) - \
                (0.01 * player.entropies[i].sum())  # 更新theta 公式(9)

        """
        每个线程和环境交互到一定量的数据后，就计算在自己线程里的神经网络损失函数的梯度，
        但是这些梯度却并不更新自己线程里的神经网络，而是去更新公共的神经网络。
        也就是n个线程会独立的使用累积的梯度分别更新公共部分的神经网络模型参数。
        每隔一段时间，线程会将自己的神经网络的参数更新为公共神经网络的参数，进而指导后面的环境交互。
        """
        player.model.zero_grad()
        # policy_loss + 0.5 * value_loss即为loss
        if rank==1:
            writer.add_scalar('VLoss/train', value_loss,
                          shared_model.training_steps.weight.data.cpu().numpy()[0])
            writer.add_scalar('PLoss/train', policy_loss,
                          shared_model.training_steps.weight.data.cpu().numpy()[0])
        (policy_loss + 0.5 * value_loss).backward()  # 计算该worder的损失函数梯度
        ensure_shared_grads(player.model, shared_model)  # 该worker将自己的参数传给公用的模型
        optimizer.step()  # optimizer为shared_model的  step()将参数更新值施加到shared_model的parameters 上
        player.clear_actions()
        if shared_model.training_steps.weight.data.cpu().numpy() > args.training_steps:
            print('num of success={0},training episodes={1},success_rate={2}'.format(shared_model.success_num, shared_model.done_nums, shared_model.success_num/shared_model.done_nums))
            break
