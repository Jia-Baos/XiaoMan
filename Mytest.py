import torch
import torch.nn as nn

x = torch.tensor([2, 3, 4], dtype=torch.float, requires_grad=True)
print(x)
y = x * 2
# while y.norm() < 1000:
#     y = y * 2
print(y)

y.backward(torch.ones_like(y))
print(x.grad)

tensor_x = torch.tensor([
    [1.732, 0.541],
    [1.732, 0.541],
    [1.732, 0.541]
], dtype=torch.float, requires_grad=True)

tensor_label = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0]
], dtype=torch.float, requires_grad=True)

softmax = nn.Softmax(dim=1)
tensor_y = softmax(tensor_x)
tensor_y.retain_grad()
tensor_label.retain_grad()
print(tensor_y)
loss = nn.CrossEntropyLoss()
cur_loss = loss(tensor_y, tensor_label)
print(cur_loss)

cur_loss.backward()
print(tensor_x.grad)


x = torch.rand(3, dtype=torch.float, requires_grad=True)
y = torch.rand(3, dtype=torch.float, requires_grad=True)
z = torch.rand(3, dtype=torch.float, requires_grad=True)
print(x)
print(y)
print(z)
result1 = torch.dot(x + y, z)
print(result1)

result2 = x + y
result2.backward(z)
print(x.grad)
