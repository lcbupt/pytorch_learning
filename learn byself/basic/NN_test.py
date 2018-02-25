import torch
from torch import nn,optim
from torch.autograd import Variable
from torch.utils.data import dataloader
from torchvision import datasets,transforms
import simplenn

batch_size = 64
learning_rate = 1e-2
num_epoches = 20

data_tf = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5],[0.5])])

train_dataset = datasets.MNIST(
    root='./MNIST_data',train=True,transform=data_tf,download=True)
test_dataset = datasets.MNIST(root='./MNIST_data',train=False,transform=data_tf)
train_loader = dataloader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = dataloader(test_dataset, batch_size=batch_size, shuffle=True)

model = simplenn.Net(28*28,300,100,10)
if torch.cuda.is_available():
    model = model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr=learning_rate)

model.eval()
eval_loss = 0
eval_acc = 0
print(eval_acc.size(0))

for data in test_loader:
    img,label = data
    img = img.view(img.size(0),-1)
    if torch.cuda.is_available():
        img = Variable(img,volatile=True).cuda()
        label = Variable(label,volatile=True).cuda()
    else:
        img = Variable(img,volatile=True)
        label= Variable(label,volatile=True)

    out = model(img)
    loss = criterion(out, label)
    eval_loss += loss.data[0] * label.size(0)
    _,pred = torch.max(out, 1)
    num_correct = (pred==label).sum()
    eval_acc += num_correct.data[0]

print('Test loss:{:.6f},ACC:{:.6f}'.format(eval_loss/(len(test_dataset)),
                                           eval_acc/(len(test_dataset))))