import torch
from torch.autograd.variable import *
from torch import nn
from matplotlib import pyplot as plt
from numpy import *



def make_features(x):
    x = x.unsqueeze(1)
    return torch.cat([x**i for i in range(1,4)],1)

w_target = torch.FloatTensor([0.5,3,2.4]).unsqueeze(1)
b_target = torch.FloatTensor([0.9])


def f(x):
    return x.mm(w_target)+b_target[0]


def get_batch(batch_size = 32):
    random = torch.randn(batch_size)
    x = make_features(random)
    y = f(x)

    if torch.cuda.is_available():
        return Variable(x).cuda(), Variable(y).cuda()
    else:
        return Variable(x),Variable(y)


class poly_model(nn.Module):
    def __init__(self):
        super(poly_model,self).__init__()
        self.poly = nn.Linear(3,1)


    def forward(self,x):
        out = self.poly(x)
        return out


if torch.cuda.is_available():
    model = poly_model().cuda()
else:
    model = poly_model()


criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr = 1e-3)

epoch = 0
while True:
    batch_x,batch_y = get_batch()

    # forward
    output = model(batch_x)
    loss = criterion(output,batch_y)
    print_loss = loss.data[0]


    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    epoch = epoch+1
    if print_loss < 1e-3:
        break

for param in model.parameters():
    print(param)

model.eval()
random = torch.randn(32)
batch_x = make_features(random)
batch_y = f(batch_x)


plt.plot(random.numpy(),batch_y.numpy(),'ro',label='Original data')
#plt.plot(random.numpy(),predict.data.numpy(),label='Fitting Line')
plt.show()


