import torch
import numpy as np

np_data = np.arange(6).reshape((2, 3))
torch_data = torch.from_numpy(np_data)
tensor2array = torch_data.numpy()
print(
    '\nnumpy array', np_data,
    '\ntorch tensor', torch_data,
    '\ntensor to array', tensor2array,
)
data = [-1, -2, 1, 2]
tensor = torch.FloatTensor(data)  # 转换成32位浮点tensor
print(
    '\nabs',
    '\nnumpy:', np.abs(data),
    '\ntorch:', torch.abs(tensor)
)
print(
    '\nsin',
    '\nnumpy:', np.sin(data),
    '\ntorch:', torch.sin(tensor)
)
print(
    '\nmean',
    '\nnumpy:', np.mean(data),
    '\ntorch:', torch.mean(tensor)
)

data = [[1, 2], [3, 4]]
tensor = torch.FloatTensor(data)
print(
    '\nmatrix multiplication',  # 矩阵乘法
    '\nnumpy:', np.matmul(data, data),
    '\ntorch:', torch.mm(tensor, tensor)
)
print(
    '\nmatrix dot',  # 矩阵点乘不求和
    '\nnumpy:', np.multiply(data, data),
    '\ntorch:', torch.mul(tensor, tensor)
)
print(
    '\nmatrix multiplication',  # 矩阵点乘不求和
    '\nnumpy:', np.array(data)*np.array(data),
    '\ntorch:', tensor * tensor
)
