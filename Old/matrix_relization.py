import torch
import torch.nn as nn

X = torch.tensor([
    [1, 1, 0, 0],
    [0, 0, 1, 1],
    [1, 0, 0, 1]
], dtype=torch.float)

W1 = torch.tensor([
    [0.5, 0.7, 0.6],
    [0.5, 0.4, 0.6],
    [0.4, 0.9, 1.0],
    [0.4, 0.8, 0.5]
], dtype=torch.float)

W2 = torch.tensor([
    [0.6, 0.9, 0.1],
    [0.1, 0.0, 0.8],
    [0.8, 0.9, 1.0],
], dtype=torch.float)

W3 = torch.tensor([
    [0.8, 0.5],
    [0.8, 0.1],
    [0.6, 0.1],
], dtype=torch.float)

y = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0]
], dtype=torch.float)

for epoch in range(500):
    print("the epoch: {}".format(epoch))
    # forward: step1
    sigmoid = nn.Sigmoid()
    S1 = torch.mm(X, W1)
    Z1 = sigmoid(S1)
    # print(S1)
    # print(Z1)

    # forward: step2
    S2 = torch.mm(Z1, W2)
    Z2 = sigmoid(S2)
    # print(S2)
    # print(Z2)

    # forward: step3
    softmax = nn.Softmax(dim=1)
    O3 = torch.mm(Z2, W3)
    y_hat = softmax(O3)
    # print(O3)
    print(y_hat)

    # backward: step1
    # Z2 @ W3 = O, softmax(O) = y_hat
    g3 = torch.mm(Z2.transpose(0, 1), y_hat - y)
    # print(g3)

    # backward: step1
    # Z1 @ W2 = S2, sigmoid(S2) = Z2
    sigmoid = nn.Sigmoid()
    g2_1 = torch.mm(y_hat - y, W3.transpose(0, 1))
    g2_2 = torch.mul(sigmoid(S2), torch.ones_like(S2) - sigmoid(S2))
    g2_3 = torch.mul(g2_1, g2_2)
    g2 = torch.mm(Z1.transpose(0, 1), g2_3)
    # print(g2)

    # backward: step1
    # X1 @ W1 = S1, sigmoid(S1) = Z1
    g1_1 = torch.mm(g2_3, W2.transpose(0, 1))
    g1_2 = torch.mul(sigmoid(S1), torch.ones_like(S1) - sigmoid(S1))
    g1_3 = torch.mul(g1_1, g2_2)
    g1 = torch.mm(X.transpose(0, 1), g1_3)
    # print(g1)

    W1 = W1 - g1
    W2 = W2 - g2
    W3 = W3 - g3
