import sys
import os
import torch
import numpy as np
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms as tfs
from datetime import datetime
import matplotlib.pyplot as plt
import random
from torchvision import models

sys.path.append('/home/xiaozhi/Documents/PycharmProjects/pytorch learning/learn byself/object segmentation/FCN')

print('********开始学习数据集********')
'''
# 图片可视化了解数据集
im_show1 = Image.open('./data/VOCdevkit/VOC2012/JPEGImages/2007_005210.jpg')
label_show1 = Image.open('./data/VOCdevkit/VOC2012/SegmentationClass/2007_005210.png').convert('RGB')
im_show2 = Image.open('./data/VOCdevkit/VOC2012/JPEGImages/2007_000645.jpg')
label_show2 = Image.open('./data/VOCdevkit/VOC2012/SegmentationClass/2007_000645.png').convert('RGB')
_, figs = plt.subplots(2, 2, figsize=(10, 8))
figs[0][0].imshow(im_show1)
figs[0][0].axes.get_xaxis().set_visible(False)
figs[0][0].axes.get_yaxis().set_visible(False)
figs[0][1].imshow(label_show1)
figs[0][1].axes.get_xaxis().set_visible(False)
figs[0][1].axes.get_yaxis().set_visible(False)
figs[1][0].imshow(im_show2)
figs[1][0].axes.get_xaxis().set_visible(False)
figs[1][0].axes.get_yaxis().set_visible(False)
figs[1][1].imshow(label_show2)
figs[1][1].axes.get_xaxis().set_visible(False)
figs[1][1].axes.get_yaxis().set_visible(False)
print('im_show1_size:', im_show1.size)
print('im_show2_size:', im_show2.size)
plt.show()
'''
print('********数据集学习完毕完毕********\n')


##################################################################################################
print('********开始学习图片Crop********')
# 图片读取和预处理函数
# 图片读取函数
voc_root = './data/VOCdevkit/VOC2012'


def read_images(root=voc_root, train=True):
    txt_fname = root + '/ImageSets/Segmentation/' + ('train.txt' if train else 'val.txt')
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    data = [os.path.join(root, 'JPEGImages', i+'.jpg') for i in images]
    label = [os.path.join(root, 'SegmentationClass', i+'.png') for i in images]
    return data, label


# 图片预处理，将图片crop成固定大小
# 在前面使用卷积网络进行图片分类的任务中，我们通过 resize 的办法对图片进行了缩放，使得他们的大小相同，
# 但是这里会遇到一个问题，对于输入图片我们当然可以 resize 成任意我们想要的大小，但是 label 也是一张图片，
# 且是在 pixel 级别上的标注，所以我们没有办法对 label 进行有效的 resize 似的其也能达到像素级别的匹配，
# 所以为了使得输入的图片大小相同，我们就使用 crop 的方式来解决这个问题，也就是从一张图片中 crop 出固定大小的区域，
# 然后在 label 上也做同样方式的 crop

def rand_crop(data, label, height, width):
    # data is PIL.Image object
    # label is PIL.Image object
    # print('width,height:', data.size[0],data.size[1])
    range_h = data.size[1] - height
    range_w = data.size[0] - width
    y0 = random.randint(0,range_h)
    x0 = random.randint(0,range_w)
    x1 = x0 + width
    y1 = y0 + height
    # print('range_w,range_h:', range_w,range_h)
    # print('x0,y0,x1,y1:',x0,y0,x1,y1)
    box = (x0,y0,x1,y1)    # left,upper,right,lower
    data = data.crop(box)
    label = label.crop(box)
    return data, label

'''
# 验证一下crop的效果
_, figs = plt.subplots(2, 2, figsize=(10, 8))
crop_im1, crop_label1 = rand_crop(im_show1, label_show1, 200, 300)
figs[0][0].imshow(crop_im1)
figs[0][1].imshow(crop_label1)
figs[0][0].axes.get_xaxis().set_visible(False)
figs[0][1].axes.get_yaxis().set_visible(False)

crop_im2, crop_label2 = rand_crop(im_show1, label_show1, 200, 300)
figs[1][0].imshow(crop_im2)
figs[1][1].imshow(crop_label2)
figs[1][0].axes.get_xaxis().set_visible(False)
figs[1][1].axes.get_yaxis().set_visible(False)
plt.show()
'''
print('********Crop效果验证完毕********\n')


##################################################################################################
print('********开始学习label图片转换成索引矩阵********')
# VOC图片类别
classes = ['background','aeroplane','bicycle','bird','boat',
           'bottle','bus','car','cat','chair','cow','diningtable',
           'dog','horse','motorbike','person','potted plant',
           'sheep','sofa','train','tv/monitor']

# 每种类别的RGB值
colormap = [[0,0,0],[128,0,0],[0,128,0], [128,128,0], [0,0,128],
            [128,0,128],[0,128,128],[128,128,128],[64,0,0],[192,0,0],
            [64,128,0],[192,128,0],[64,0,128],[192,0,128],
            [64,128,128],[192,128,128],[0,64,0],[128,64,0],
            [0,192,0],[128,192,0],[0,64,128]]

cm2lbl = np.zeros(256**3)   # 每个像素点有 0 ~ 255 的选择，RGB 三个通道
for i,cm in enumerate(colormap):
    cm2lbl[(cm[0]*256+cm[1])*256+cm[2]] = i     # 建立索引

# print('索引：', cm2lbl[0:1000])


# 根据索引将label图像转换成label矩阵
def image2label(im):
    data = np.array(im, dtype='int32')
    idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
    return np.array(cm2lbl[idx], dtype='int64')     # 根据索引得到label矩阵


# 验证索引建立效果
label_im = Image.open('./data/VOCdevkit/VOC2012/SegmentationClass/2007_000033.png').convert('RGB')
label = image2label(label_im)
print('label图片转换成索引矩阵：\n', label[150:160, 240:250])
print('********label图片转换成索引矩阵验证完毕********\n')


##################################################################################################
print('********开始图片数据综合预处理********')


# 综合数据预处理函数
def img_transforms(im, label, crop_size):
    im, label = rand_crop(im, label, *crop_size)    # 随机裁剪
    im_tfs = tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    im = im_tfs(im) # 转化成tensor并标准化
    label = image2label(label)  # 将label图片抓花城索引矩阵
    label = torch.from_numpy(label) # 将矩阵转化为tensor
    return im, label


# 定义自己的VOC数据加载类
class VOCSegDataset(Dataset):

    def __init__(self, train, crop_size, transforms):
        self.crop_size = crop_size
        self.transforms = transforms
        data_list, label_list = read_images(train=train)
        self.data_list = self._filter(data_list)
        self.label_list = self._filter(label_list)
        print('Read ' + str(len(self.data_list)) + ' images')

    # 过滤掉图片大小小于 crop 大小的图片
    def _filter(self, images):
        return [im for im in images if (Image.open(im).size[1] >= self.crop_size[0] and
                                        Image.open(im).size[0] >= self.crop_size[1])]

    def __getitem__(self, idx):
        img = self.data_list[idx]
        label = self.label_list[idx]
        img = Image.open(img)
        label = Image.open(label).convert('RGB')
        img, label = self.transforms(img, label, self.crop_size)
        return img, label

    def __len__(self):
        return len(self.data_list)


# 实例化数据集
input_shape = (320, 480)
voc_train = VOCSegDataset(True, input_shape, img_transforms)
voc_test = VOCSegDataset(False, input_shape, img_transforms)

train_data = DataLoader(voc_train, 64, shuffle=True, num_workers=4)
valid_data = DataLoader(voc_test, 128, num_workers=4)
print('********数据综合预处理完毕********\n')
##################################################################################################
print('********开始学习bilinear kernel对转置卷积初始化权重********')


# 定义 bilinear kernel
def bilinear_kernel(in_channels, out_channels, kernel_size):
    # return a bilinear filter tensor
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight)


# 测试bilinear kernel转置卷积的效果
# 可以看到通过双线性的 kernel 进行转置卷积，图片的大小扩大了一倍，但是图片看上去仍然非常的清楚，
# 所以这种方式的上采样具有很好的效果
x = Image.open('./data/VOCdevkit/VOC2012/JPEGImages/2007_005210.jpg')
x = np.array(x)
plt.imshow(x)
print('x_shape:', x.shape)
# plt.show()
# print('x1：', torch.from_numpy(x.astype('float32')))
# print('x2：', torch.from_numpy(x.astype('float32')).permute(2,0,1))
# print('x3：', torch.from_numpy(x.astype('float32')).permute(2,0,1).unsqueeze(0))
x = torch.from_numpy(x.astype('float32')).permute(2, 0, 1).unsqueeze(0)


# 定义转置卷积，卷积的逆过程
conv_trans = nn.ConvTranspose2d(3, 3, 4, 2, 1)

# 将其定义为 bilinear kernel
conv_trans.weight.data = bilinear_kernel(3, 3, 4)

# print('y1：', conv_trans(Variable(x)).data)
# print('y2：', conv_trans(Variable(x)).data.squeeze())
# print('y3：', conv_trans(Variable(x)).data.squeeze().permute(1, 2, 0))
# print('y4：', conv_trans(Variable(x)).data.squeeze().permute(1, 2, 0).numpy())
y = conv_trans(Variable(x)).data.squeeze().permute(1, 2, 0).numpy()
plt.imshow(y.astype('uint8'))
print('y_shape:', y.shape)
# plt.show()
print('********bilinear kernel对转置卷积初始化权重学习完毕********\n')
##################################################################################################

print('********开始定义模型********')
# 定义模型

# 使用预训练的resnet 34
pretrained_net = models.resnet34(pretrained=True)
num_classes = len(classes)


class fcn(nn.Module):
    def __init__(self, num_classes):
        super(fcn, self).__init__()

        self.stage1 = nn.Sequential(*list(pretrained_net.children())[:-4]) # 第一段
        self.stage2 = list(pretrained_net.children())[-3] # 第二段
        self.stage3 = list(pretrained_net.children())[-2] # 第三段

        self.scores1 = nn.Conv2d(512, num_classes, 1)
        self.scores2 = nn.Conv2d(256, num_classes, 1)
        self.scores3 = nn.Conv2d(128, num_classes, 1)
        
        self.upsample_8x = nn.ConvTranspose2d(num_classes, num_classes, 16, 8, 4, bias=False)
        self.upsample_8x.weight.data = bilinear_kernel(num_classes, num_classes, 16)    # 使用双线性 kernel
        
        self.upsample_4x = nn.ConvTranspose2d(num_classes, num_classes, 4, 2, 1, bias=False)
        self.upsample_4x.weight.data = bilinear_kernel(num_classes, num_classes, 4)     # 使用双线性 kernel
        
        self.upsample_2x = nn.ConvTranspose2d(num_classes, num_classes, 4, 2, 1, bias=False)   
        self.upsample_2x.weight.data = bilinear_kernel(num_classes, num_classes, 4)     # 使用双线性 kernel

    def forward(self, x):
        x = self.stage1(x)
        s1 = x  # 1/8

        x = self.stage2(x)
        s2 = x  # 1/16
        
        x = self.stage3(x)
        s3 = x  # 1/32
        
        s3 = self.scores1(s3)
        s3 = self.upsample_2x(s3)
        s2 = self.scores2(s2)
        s2 = s2 + s3
        
        s1 = self.scores3(s1)
        s2 = self.upsample_4x(s2)
        s = s1 + s2

        s = self.upsample_8x(s)
        return s


net = fcn(num_classes)


# 定义一些语义分割常用的指标
def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


def label_accuracy_score(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc


# 定义 loss 和 optimizer
criterion = nn.NLLLoss2d()
optimizer = torch.optim.SGD(net.parameters(), lr=1e-2, weight_decay=1e-4)
print('********模型定义完成********\n')
##################################################################################################
print('********开始训练********')
for e in range(80):
    train_loss = 0
    train_acc = 0
    train_acc_cls = 0
    train_mean_iu = 0
    train_fwavacc = 0
    
    prev_time = datetime.now()
    net = net.train()
    for data in train_data:
        im = Variable(data[0])
        label = Variable(data[1])

        # forward
        out = net(im)
        out = F.log_softmax(out)     # (b, n, h, w)
        loss = criterion(out, label)
        print('**********!')

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.data[0]
        
        label_pred = out.max(dim=1)[1].data.cpu().numpy()
        label_true = label.data.cpu().numpy()
        for lbt, lbp in zip(label_true, label_pred):
            acc, acc_cls, mean_iu, fwavacc = label_accuracy_score(lbt, lbp, num_classes)
            train_acc += acc
            train_acc_cls += acc_cls
            train_mean_iu += mean_iu
            train_fwavacc += fwavacc

    # 验证模型
    net = net.eval()
    eval_loss = 0
    eval_acc = 0
    eval_acc_cls = 0
    eval_mean_iu = 0
    eval_fwavacc = 0
    for data in valid_data:
        im = Variable(data[0], volatile=True)
        label = Variable(data[1], volatile=True)
        # forward
        out = net(im)
        out = F.log_softmax(out)
        loss = criterion(out, label)
        eval_loss += loss.data[0]
        
        label_pred = out.max(dim=1)[1].data.cpu().numpy()
        label_true = label.data.cpu().numpy()
        for lbt, lbp in zip(label_true, label_pred):
            acc, acc_cls, mean_iu, fwavacc = label_accuracy_score(lbt, lbp, num_classes)
            eval_acc += acc
            eval_acc_cls += acc_cls
            eval_mean_iu += mean_iu
            eval_fwavacc += fwavacc
        
    cur_time = datetime.now()
    h, remainder = divmod((cur_time - prev_time).seconds, 3600)
    m, s = divmod(remainder, 60)
    epoch_str = ('Epoch: {}, Train Loss: {:.5f}, Train Acc: {:.5f}, Train Mean IU: {:.5f}, \
Valid Loss: {:.5f}, Valid Acc: {:.5f}, Valid Mean IU: {:.5f} '.format(
        e, train_loss / len(train_data), train_acc / len(voc_train), train_mean_iu / len(voc_train),
        eval_loss / len(valid_data), eval_acc / len(voc_test), eval_mean_iu / len(voc_test)))
    time_str = 'Time: {:.0f}:{:.0f}:{:.0f}'.format(h, m, s)
    print(epoch_str + time_str)

# net.load_state_dict(torch.load('./fcn.pth'))
print('*********训练结束！*********\n')
##################################################################################################
'''
# 验证结果可视化
net = net.eval()

# 定义预测函数
cm = np.array(colormap).astype('uint8')

def predict(im, label): # 预测结果
    im = Variable(im.unsqueeze(0))
    out = net(im)
    pred = out.max(1)[1].squeeze().cpu().data.numpy()
    pred = cm[pred]
    return pred, cm[label.numpy()]
    
_, figs = plt.subplots(6, 3, figsize=(12, 10))
for i in range(6):
    test_data, test_label = voc_test[i]
    pred, label = predict(test_data, test_label)
    figs[i, 0].imshow(Image.open(voc_test.data_list[i]))
    figs[i, 0].axes.get_xaxis().set_visible(False)
    figs[i, 0].axes.get_yaxis().set_visible(False)
    figs[i, 1].imshow(label)
    figs[i, 1].axes.get_xaxis().set_visible(False)
    figs[i, 1].axes.get_yaxis().set_visible(False)
    figs[i, 2].imshow(pred)
    figs[i, 2].axes.get_xaxis().set_visible(False)
    figs[i, 2].axes.get_yaxis().set_visible(False)

'''