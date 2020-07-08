import numpy as np
import pandas as pd
import time

np.random.seed(2)

N_STATES = 6  #the length of the 1 dimensional world起点距宝藏的距离
ACTIONS = ['left', 'right'] #available actions
EPSILON = 0.9   #greedy police  #0.9选择最优动作，0.1随机动作
ALPHA = 0.1     #learning rate      学习效率
LAMBDA = 0.9    #discount factor    递减度，对未来奖励的衰减
MAX_EPISODES = 13   #maximum episodes   最多13回合
FRESH_TIME = 0.3  #fresh time for one move    走一步的时间

#行为a在s状态的值：Q(s,a)
def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))), #q_table initial values
        columns=actions,    #actions' name
    )
    return table

def choose_action(state, q_table):
    state_actions = q_table.iloc[state, :]  # iloc 是用基于整数的下标来进行数据定位/选择
    
    #0.1的概率随机
    #numpy.random.uniform(low, high, size) 功能：从一个均匀分布[low, high)中随机采样。默认为0，1

    if (np.random.uniform() > EPSILON or (state_actions.all() == 0)):
        action_name=np.random.choice(ACTIONS)
    else:
        action_name = state_actions.idxmax()    #获取最大索引值
    return action_name

def get_env_feedback(S, A):
    #R: reward。 A: action。 S: 这个state  S_: 下个state
    if A == 'right':    #move right
        if S == N_STATES - 2:   #terminate
            S_ = "terminal"
            R = 1
        else:
            S_ = S + 1
            R = 0
    else:   #move left
        R = 0
        if S == 0:
            S_ = S  #reach the wall
        else:
            S_ = S - 1
    return S_, R
    

def update_env(S, episode, step_counter):
    # This is how environment be updated
    env_list = ['-']*(N_STATES-1) + ['T']   # '---------T' our environment
    if S == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (
            episode+1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                ', end='')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)


def rl():
    q_table = build_q_table(N_STATES, ACTIONS)  # 初始 q table
    for episode in range(MAX_EPISODES):     # 回合
        step_counter = 0
        S = 0   # 回合初始位置
        is_terminated = False   # 是否回合结束
        update_env(S, episode, step_counter)    # 环境更新
        while not is_terminated:

            A = choose_action(S, q_table)   # 选行为
            S_, R = get_env_feedback(S, A)  # 实施行为并得到环境的反馈
            q_predict = q_table.loc[S, A]    # 估算的(状态-行为)值
            if S_ != 'terminal':
                # 实际的(状态-行为)值 (回合没结束)
                q_target = R + LAMBDA * q_table.iloc[S_, :].max()
            else:
                q_target = R  # 实际的(状态-行为)值 (回合结束)
                is_terminated = True    # terminate this episode
            #  loc 可以通过行号和行标签进行索引，比如 df.loc['a'] , df.loc[1] 
            q_table.loc[S, A] += ALPHA * (q_target - q_predict)  # q_table 更新
            S = S_  # 探索者移动到下一个 state

            update_env(S, episode, step_counter+1)  # 环境更新

            step_counter += 1
    return q_table

if __name__ == "__main__":
    q_table = rl()
    print('\r\nQ-table:\n')
    print(q_table)



