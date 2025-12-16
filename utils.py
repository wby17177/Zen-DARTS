import os
import numpy as np
import torch
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable


# 计算均值
class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


# 生成一个length的正方形遮挡（不能超过图像边框）
class Cutout(object):

    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)  # 取图像的高h和宽w
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)  # 取0-h的整数
        x = np.random.randint(w)  # 取0-w的整数

        y1 = np.clip(y - self.length // 2, 0, h)  # 从mask的中点寻找y1点，并保证不超过图像的最大高度
        y2 = np.clip(y + self.length // 2, 0, h)  # 同理寻找y2
        x1 = np.clip(x - self.length // 2, 0, w)  # 寻找x1
        x2 = np.clip(x + self.length // 2, 0, w)  # 寻找x2

        mask[y1: y2, x1: x2] = 0.  # 将mask对应的位置置零，其他位置为1
        mask = torch.from_numpy(mask)  # 将numpy的数组变量转变为Tensor变量
        mask = mask.expand_as(img)  # 将mask tensor扩张为和img一个size
        img *= mask
        return img

#
def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))    # .eq为比较

    res = []
    # 循环计算topk正确率，k=1时，是预测最大概率的样本的正确率，k=2是第二。
    for k in topk:
        # 计算预测正确的样本数,此处可能一个tensor在两个batch中，所以此处需要使用.contiguous()
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))    # 计算正确率
    return res


# 图像增强
def _data_transforms_cifar10(args):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 裁剪
        transforms.RandomHorizontalFlip(),  # 随机反转
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),  # 归一化处理
    ])
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return train_transform, valid_transform


# 统计参数的量
def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name) / 1e6


# 保存当前checkpoint——checkpoint指保持模型及其参数，以及优化器参数和loss，epoch等
def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    # 判定是否为最佳模型，若是则单独复制保存替换上一份最佳模型
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)


def save(model, model_path):
    torch.save(model.state_dict(), model_path)


def load(model, model_path):
    model.load_state_dict(torch.load(model_path), strict=False)


# 以drop_prob为概率取0，其余概率取1，对路径进行随机丢弃
def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob
        mask = Variable(
            torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))  # 根据x行数生成对应的Mask tensor，有几行可以生成几个
        x.div_(keep_prob)
        x.mul_(mask)
    return x


# 创建文件并复制一些文件
def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    # print('Experiment dir : {}'.format(path))

    # if scripts_to_save is not None:
        # os.mkdir(os.path.join(path, 'scripts'))
         # for script in scripts_to_save:
            # dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            # shutil.copyfile(script, path)  # 复制script到dstfile
