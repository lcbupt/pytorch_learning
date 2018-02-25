import torch
import numpy as np
from torch.autograd.variable import *
from torch.utils.data import *
import pandas as pd

'''
#张量Tensor
a = torch.Tensor([[2,3],[4,8],[7,9]])
print('a is:{}'.format(a))
print('a size is {}'.format(a.size()))

c = torch.zeros((3,2))
print('zero tensor:{}'.format(c))

d = torch.randn((3,2))
print('normal randon is :{}'.format(d))

a[0,1] = 100
print('changed a is :{}'.format(a))

b = torch.LongTensor([[2,3],[4,8],[7,9]])
print('b is: {}'.format(b))

numpy_b = b.numpy()
print('numpy_b:\n{}'.format(numpy_b))

e = np.array([[2,3],[4,6]])
torch_e = torch.from_numpy(e)
print('torch_e:\n{}'.format(torch_e))

f_torch_e = torch_e.float()
print('f_torch_e:\n{}'.format(f_torch_e))
'''

'''
#变量Variable
x = Variable(torch.Tensor([1]), requires_grad = True)
print('x:{}'.format(x))
w = Variable(torch.Tensor([2]), requires_grad = True)
print('w:{}'.format(w))
b = Variable(torch.Tensor([3]), requires_grad = True)
print('b:{}'.format(b))

y = w * x + b
y.backward()
print('x_grad:{}'.format(x.grad))
print('w_grad:{}'.format(w.grad))
print('b_grad:{}'.format(b.grad))


x = torch.randn(3)
x = Variable(x, requires_grad = True)

y = x*2
print('y:{}'.format(y))

y.backward(torch.FloatTensor([1,0.1,0.01]))
print('x_grad:{}'.format(x.grad))
'''

a = 100
print(type(a))
print(a+100)

class myDataset(Dataset):
    def __init__(self,csv_file,txt_file,root_dir,other_file):
        self.csv_data = pd.read_csv(csv_file)
        with open(txt_file,'r') as f:
            data_list = f.readlines()
        self.txt_data = data_list
        self.root_dir = root_dir

        def __len__(self):
            return len(self.csv_data)

        def __getitem__(self, idx):
            data = (self.csv_data[idx], self.txt_data[idx])
            return data