import torch
import torch.nn as nn


class MyRNN(nn.Module):
    def __init__(self, in_features=6, hidden_size=2, n_classes=2):
        super(MyRNN, self).__init__()
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.n_classes = n_classes
        self.linear1 = nn.Linear(self.in_features + self.hidden_size, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, self.n_classes)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        # x -> [batch, seq_len, in_features]
        pre_hidden_state = torch.zeros((x.shape[0], x.shape[1], self.hidden_size), dtype=torch.float)

        for i in range(x.shape[1]):
            output = torch.cat((pre_hidden_state, x), dim=2)
            print("cat's size: ", output.size())
            hidden_state = self.linear1(output)
            hidden_state = self.tanh(hidden_state)
            print("hidden_state's size: ", hidden_state.size())

            # 修正下一轮所使用的hidden_state
            pre_hidden_state = hidden_state

            output = self.linear2(hidden_state)
            if self.n_classes == 1:
                output = self.sigmoid(output)
                print("final output's size: ", output.size())
            else:
                output = self.softmax(output)
                print("final output's size: ", output.size())

        return output, hidden_state


if __name__ == '__main__':
    """
    此处错误可参考连接： https://blog.csdn.net/Mr_wuliboy/article/details/90520011
    关于这个错误仔细思考了以下，原因在于forward的返回结果有两个————output、hidden state
    所以要用 output, hidden_state = model(x) 的形式来接受结果
    如果直接 y = model(x) 则会报错，因为此时返回的是一个元组
    哈哈哈哈哈，很有意思的一件事哈
    """

    x = torch.rand((4, 6, 8), dtype=torch.float)
    print("input's size: ", x.size())
    model = MyRNN(in_features=x.shape[2], hidden_size=2, n_classes=2)
    output, hidden_state = model(x)
    print(output)
    print(type(output))
    print(hidden_state)
    print(type(hidden_state))


