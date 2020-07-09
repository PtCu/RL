from maze_env import Maze
from RL_brain import DQN


def run_maze():
    step = 0  # 用来控制什么时候学习
    for episode in range(300):
        # 初始化环境
        observation = env.reset()

        while True:
            # 刷新环境
            env.render()

            # DQN根据观测值选择行为
            action = RL.choose_action(observation)

            # 环境根据行为给出下一个state,reward,是否终止
            observation_, reward, done = env.step(action)

            # DQN存储记忆
            RL.store_transition(observation, action, reward, observation_)

            # 控制学习起始时间和频率（先积累一些记忆在开始学习）
            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # 将下一个state_变为下次循环的state
            observation = observation_

            # 如果终止就跳出循环
            if done:
                break
            step += 1  # 总步数
    # end of game
    print('game over')
    env.destroy()


if __name__ == "__main__":
    # maze game
    env = Maze()
    RL = DQN()
    env.after(100, run_maze)
    env.mainloop()
