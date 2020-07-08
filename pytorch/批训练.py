import torch
import torch.utils.data as Data
torch.manual_seed(1)  # reproducible

BATCH_SIZE = 5  # 批训练的数据个数

x = torch.linspace(1, 10, 10)
y = torch.linspace(10, 1, 10)

# 先转换成 torch 能识别的 Dataset
torch_dataset = Data.TensorDataset(x, y)

loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
)


def show_batch():
    for epoch in range(3):   # train entire dataset 3 times
        for step, (batch_x, batch_y) in enumerate(loader):  # for each training step
            # train your data...
            print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
                  batch_x.numpy(), '| batch y: ', batch_y.numpy())

if __name__ == '__main__':
    show_batch()
