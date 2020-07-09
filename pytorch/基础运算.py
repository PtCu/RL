# import torch
# import numpy as np

# np_data = np.arange(6).reshape((2, 3))
# torch_data = torch.from_numpy(np_data)
# tensor2array = torch_data.numpy()
# print(
#     '\nnumpy array', np_data,
#     '\ntorch tensor', torch_data,
#     '\ntensor to array', tensor2array,
# )
# data = [-1, -2, 1, 2]
# tensor = torch.FloatTensor(data)  # 转换成32位浮点tensor
# print(
#     '\nabs',
#     '\nnumpy:', np.abs(data),
#     '\ntorch:', torch.abs(tensor)
# )
# print(
#     '\nsin',
#     '\nnumpy:', np.sin(data),
#     '\ntorch:', torch.sin(tensor)
# )
# print(
#     '\nmean',
#     '\nnumpy:', np.mean(data),
#     '\ntorch:', torch.mean(tensor)
# )

# data = [[1, 2], [3, 4]]
# tensor = torch.FloatTensor(data)
# print(
#     '\nmatrix multiplication',  # 矩阵乘法
#     '\nnumpy:', np.matmul(data, data),
#     '\ntorch:', torch.mm(tensor, tensor)
# )
# print(
#     '\nmatrix dot',  # 矩阵点乘不求和
#     '\nnumpy:', np.multiply(data, data),
#     '\ntorch:', torch.mul(tensor, tensor)
# )
# print(
#     '\nmatrix multiplication',  # 矩阵点乘不求和
#     '\nnumpy:', np.array(data)*np.array(data),
#     '\ntorch:', tensor * tensor
# )
# a = 80 * 0.4 + 85 * 1 + 95 * 0.5 + 83 * 3 + 90 * 3 + 95 * 3 + 74 * 0.4 + 98.6 * 1.5 + 94 * 1 + 95 * 2 + 83 * 1 + 89 * 1 + 81 * 3 + 86 * 2.5 + 83 * 0.5 + 84 * 5 + 75 * 0.4 + 82 * \
#     2 + 87 * 2 + 82 * 2 + 86 * 4 + 88 * 2 + 92 * 2 + 80 * 4 + 70 * 2 + 81 * 3 + 83 * 2 + \
#     87 + 90 + 91 + 89 + 89 * 3 + 94 * 6 + 85 * 2 + \
#     (95 + 93 + 94) * 2 + 92 + 86 * 2 + 89 * 3
# b = 0.4 + 1 + 0.5 + 3 + 3 + 3 + 0.4 + 1.5 + 1 + 2 + 1 + 1 + 3 + 2.5 + 0.5 + 5 + 0.4 + 1 +\
#     2 + 2 + 2 + 4 + 2 + 2 + 4 + 2 + 3 + 2 + \
#     1 + 1 + 1+1 + 3 + 6 + 2 + \
#     3 * 2 + 1 + 2 + 3
# print(a/b)
# print(b)
a = [1, 2, 3, 4]
b = a.sample()
print(b)