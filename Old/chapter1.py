import torch
import torch.nn as nn

X = torch.tensor([
    [1, 1, 0, 0],
    [0, 0, 1, 1],
    [1, 0, 0, 1]
], dtype=torch.float, requires_grad=True)

W1 = torch.tensor([
    [0.5, 0.7, 0.6],
    [0.5, 0.4, 0.6],
    [0.4, 0.9, 1.0],
    [0.4, 0.8, 0.5]
], dtype=torch.float, requires_grad=True)

W2 = torch.tensor([
    [0.6, 0.9, 0.1],
    [0.1, 0.0, 0.8],
    [0.8, 0.9, 1.0],
], dtype=torch.float, requires_grad=True)

W3 = torch.tensor([
    [0.8, 0.5],
    [0.8, 0.1],
    [0.6, 0.1],
], dtype=torch.float, requires_grad=True)

test = torch.tensor([
    [1.732, 0.541],
    [1.732, 0.541],
    [1.732, 0.541],
], dtype=torch.float)

# forward: step1
sigmoid = nn.Sigmoid()
S1 = torch.mm(X, W1)
Z1 = sigmoid(S1)
print(S1)
print(Z1)

# forward: step2
S2 = torch.mm(Z1, W2)
Z2 = sigmoid(S2)
print(S2)
print(Z2)

# forward: step3
softmax = nn.Softmax(dim=1)
O3 = torch.mm(Z2, W3)
Y_hat = softmax(O3)
print(O3)
print(Y_hat)

# backward: step1
O3_grads = torch.autograd.grad(outputs=Y_hat, inputs=O3, grad_outputs=torch.ones_like(O3),
                               retain_graph=True, create_graph=True)
print(O3_grads)
