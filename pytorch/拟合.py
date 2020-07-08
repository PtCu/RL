import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
# x data (tensor), shape=(100, 1)
"""
torch.unsqueeze()这个函数主要是对数据维度进行扩充。
给指定位置加上维数为一的维度，比如原本有个三行的数据（3），
在0的位置加了一维就变成一行三列（1,3）。
a.squeeze(N) 就是在a中指定位置N加上一个维数为1的维度。
还有一种形式就是b=torch.squeeze(a，N) a就是在a中指定位置N加上一个维数为1的维度

"""
a = torch.linspace(-1, 1, 100)
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
b = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=0)
# noisy y data (tensor), shape=(100, 1)

# torch.rand(*sizes, out=None) → Tensor
# 返回一个张量，包含了从区间[0, 1)的均匀分布中抽取的一组随机数。张量的形状由参数sizes定义。
y = x.pow(2) + 0.2*torch.rand(x.size())

# 画图
# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):  # Module中的forward
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x


net = Net(n_feature=1, n_hidden=10, n_output=1)

optimizer = torch.optim.SGD(net.parameters(), lr=0.2)  # 传入net的所有参数，学习率
loss_func = torch.nn.MSELoss()  # 预测值和真实值的误差计算公式

for t in range(200):
    prediction = net(x)      # 喂给 net 训练数据 x, 输出预测值

    loss = loss_func(prediction, y)  # 计算两者的误差

    optimizer.zero_grad()    # 清空上一步的残余更新参数值   把所有Variable的grad成员数值变为0
    loss.backward()          # 误差反向传播, 计算参数更新值
    optimizer.step()  # 将参数更新值施加到 net 的 parameters 上

    if t % 5 == 0:
        # plot and show learning process
        plt.cla()  # Clear axis
        plt.scatter(x.data.numpy(), y.data.numpy())  # 散点图
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)  # 连续图
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(),
                 fontdict={'size': 20, 'color':  'red'})
        # plt.pause(0.1)

plt.ioff()
plt.show()
