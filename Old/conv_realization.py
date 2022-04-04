import torch
import torch.nn as nn

X = torch.tensor([[[
    [0.4, 0.7, 0.0],
    [0.3, 0.1, 0.1],
    [0.2, 0.3, 0.4],
    [0.5, 0.4, 0.7]
]]], dtype=torch.float)

W = torch.tensor([[-0.1149, -0.8085]], dtype=torch.float)
y = torch.tensor([1.0, 0.0], dtype=torch.float)

# forward: step1
Conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(2, 3), stride=1, padding=0, bias=False)
Conv1.weight.data = torch.tensor([[[
    [1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0]
]]])
S1 = Conv1(X)
print(S1)

# forward: step2
S1 = S1.view(-1, 3)
Relu = nn.ReLU()
P1 = Relu(S1)
print(P1)

# forward: step3
Maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=0)
Z = Maxpool(P1)
print(Z)

# forward: step4
linear = nn.Linear(in_features=1, out_features=2, bias=False)
linear.weight = torch.nn.Parameter(W.transpose(0, 1))
O = linear(Z)
print(O)

# forward: step5
softmax = nn.Softmax(dim=1)
y_hat = softmax(O)
print(y_hat)

# backward: step1
g1 = y_hat - y
print(g1)

# backward: step2
g2 = torch.mm(Z.transpose(0, 1), g1)
print(g2)

# backward: step3
g3_1 = torch.mm(g1, W.transpose(0, 1))
print(g3_1)

# backward: step4
g4_1 = torch.zeros_like(P1)
index = torch.argmax(P1, dim=1)
# 此处为二维数组切片
g4_1[:, index] = 1.0
g4 = torch.mm(g3_1, g4_1)
print(g4)

# backward: step5
g5 = torch.zeros_like(g4)
size = g4.size(1)
for i in range(size):
    if S1[:, i] > 0:
        g5[:, i] = g4[:, i]
print(g5)

# backward: step5
g6_1 = (g5.unsqueeze(0)).unsqueeze(0)
g6_1 = g6_1.permute(0, 1, 3, 2)
print(g6_1)
print(g6_1.size())
Conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 3), stride=1, padding=0, bias=False)
Conv2.weight.data = torch.tensor(g6_1, dtype=torch.float)
print(Conv2.weight.data)
print(Conv2.weight.data.size())
g6 = Conv2(X)
print(g6)
