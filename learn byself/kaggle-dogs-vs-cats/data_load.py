import os
import shutil


# 创建猫狗的数据库目录
os.makedirs('/home/xiaozhi/Documents/PycharmProjects/pytorch learning/learn byself/kaggle-dogs-vs-cats/data')
os.makedirs('/home/xiaozhi/Documents/PycharmProjects/pytorch learning/learn byself/kaggle-dogs-vs-cats/data/train')
os.makedirs('/home/xiaozhi/Documents/PycharmProjects/pytorch learning/learn byself/kaggle-dogs-vs-cats/data/val')


train_root = '/home/xiaozhi/Documents/PycharmProjects/pytorch learning/learn byself/kaggle-dogs-vs-cats/data/train'
dog_folder = os.path.join(train_root, 'dog')
cat_folder = os.path.join(train_root, 'cat')
os.makedirs(dog_folder)
os.makedirs(cat_folder)

val_root = '/home/xiaozhi/Documents/PycharmProjects/pytorch learning/learn byself/kaggle-dogs-vs-cats/data/val'
dog_folder = os.path.join(val_root, 'dog')
cat_folder = os.path.join(val_root, 'cat')
os.makedirs(dog_folder)
os.makedirs(cat_folder)


# 分散加载猫狗图片
data_file = os.listdir('/home/xiaozhi/Documents/PycharmProjects/pytorch learning/learn byself/kaggle-dogs-vs-cats/train_zip')
dog_file = list(filter(lambda x: x[:3]=='dog', data_file))   # 根据文件名的前三个字母区分猫狗文件
cat_file = list(filter(lambda x: x[:3]=='cat', data_file))

root = '/home/xiaozhi/Documents/PycharmProjects/pytorch learning/learn byself/kaggle-dogs-vs-cats/'  # 定义根目录
for i in range(len(dog_file)):
    pic_path = root + 'train_zip/' + dog_file[i]  # 拼接狗数据的当前存放路径
    if i < len(dog_file) * 0.9: # 90%训练，10%测试
        obj_path = train_root + '/dog/' + dog_file[i]   # 拼接狗数据的目标存放路径
    else:
        obj_path = val_root + '/dog/' + dog_file[i]
    shutil.move(pic_path, obj_path)     # 移动图片

for i in range(len(cat_file)):
    pic_path = root + 'train_zip/' + cat_file[i]  # 拼接猫数据的当前存放路径
    if i < len(dog_file)*0.9:   # 90%训练，10%测试
        obj_path = train_root + '/cat/' + cat_file[i]   # 拼接猫数据的目标存放路径
    else:
        obj_path = val_root + '/cat/' + cat_file[i]
    shutil.move(pic_path, obj_path)     # 移动图片