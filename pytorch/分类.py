import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
# 假数据
"""
torch.normal(means, std, out=None)
返回一个张量，包含从给定参数means,std的离散正态分布中抽取随机数。 
均值means是一个张量，包含每个输出元素相关的正态分布的均值。 
std是一个张量，包含每个输出元素相关的正态分布的标准差。 均值和标准差的形状不须匹配，但每个张量的元素个数须相同。
"""
n_data = torch.ones(100, 2)  # 数据的基本形态
x0 = torch.normal(2 * n_data, 1)  # 类型0 x data (tensor), shape=(100, 2)
y0 = torch.zeros(100)       # 类型0 y data (tensor), shape=(100, )
x1 = torch.normal(-2 * n_data, 1)     # 类型1 x data (tensor), shape=(100, 1)
y1 = torch.ones(100)      # 类型1 y data (tensor), shape=(100, )

# 注意 x, y 数据的数据形式是一定要像下面一样 (torch.cat 是在合并数据)
# FloatTensor = 32-bit floating 按0维拼接
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)
y = torch.cat((y0, y1),).type(torch.LongTensor)  # LongTensor = 64-bit integer

plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1],
            c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
plt.show()


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.out = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.out(x)
        return x


net = Net(n_feature=2, n_hidden=10, n_output=2)     # define the network
print(net)  # net architecture

optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
# the target label is NOT an one-hotted
loss_func = torch.nn.CrossEntropyLoss()

plt.ion()   # something about plotting

for t in range(100):
    out = net(x)                 # input x and predict based on x
    # must be (1. nn output, 2. target), the target label is NOT one-hotted
    loss = loss_func(out, y)

    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients
    optimizer.step()        # apply gradients

    if t % 2 == 0:
        # plot and show learning process
        plt.cla()
        # max(out,1):1表示每行的最大值，0表示每列的最大值 函数会返回两个tensor，第一个tensor是每行的最大值，softmax的输出中最大的是1，所以第一个tensor是全1的tensor；第二个tensor是每行最大值的索引。
        prediction = torch.max(out, 1)[1]
        pred_y = prediction.data.numpy()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[
                    :, 1], c=pred_y, s=100, lw=1, cmap='RdYlGn')  # lw粗细、大小，c颜色
        accuracy = float((pred_y == target_y).astype(
            int).sum()) / float(target_y.size)
        plt.text(1.5, -4, 'Accuracy=%.2f' %
                 accuracy, fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()
