import torch
import torch.nn as nn
from MyNet import XiaoManNet

x = torch.tensor([
    [1, 1, 0, 0],
    [0, 0, 1, 1],
    [1, 0, 0, 1]
], dtype=torch.float)

label = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
], dtype=torch.float)

model = XiaoManNet()
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1.0)

if __name__ == '__main__':
    for epoch in range(1000):
        y = model(x)
        current_loss = loss(y, label)
        print("predict value")
        print(y)
        print("current_loss: ")
        print(current_loss)

        optimizer.zero_grad()
        current_loss.backward()

        print("{}weight and bias".format(epoch))
        print(model.c3.weight)
        print(model.c2.weight)
        print(model.c1.weight)

        optimizer.step()

        print("{}weight and bias has updated".format(epoch))
        print(model.c3.weight)
        print(model.c2.weight)
        print(model.c1.weight)
