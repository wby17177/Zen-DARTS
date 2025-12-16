import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model_search import Network
from architect import Architect
from architect import ConvSeparateLoss
# 参数设置
parser = argparse.ArgumentParser("cifar")  # 解析器，包含把命令行解析成python数据类型所需的所有信息
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=80, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=100, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')   # 初始为8
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
args = parser.parse_args()

args.save = 'search-{}-{}'.format(args.save, time.strftime("2"))
# 每个实验会创建一个文件，并把所有的py文件都复制进去
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
# 该函数是输出运行日志的，stream使用指定的stream初始化，level是输出指定的级别（高于某个级别才会输出）
# format是指定的输出字符串格式，datefmt是输出的日期格式
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='2')
# logging.FileHandler为文件处理器
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
# setFormatter为设置一个格式化器
fh.setFormatter(logging.Formatter(log_format))
# 获取日志实例
logging.getLogger().addHandler(fh)

CIFAR_CLASSES = 10


def main():
    # 部署GPU
    print(torch.cuda.is_available())
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)   # 设置numpy随机种子
    torch.cuda.set_device(args.gpu)        # 部署cuda
    torch.backends.cudnn.flags(enabled=True, benchmark=True)

    # cudnn.benchmark= True    # 主要是针对torch底层进行设置，为true时，会选择计算速度最快的方式进行卷积
    torch.manual_seed(args.seed)   # 随机种子设置 (CPU) 生成随机数的种子，并返回一个torch.Generator对象。设置种子的用意是一旦固定种子，后面依次生成的随机数其实都是固定的。
    # cudnn.enabled = True    # 由于torch底层使用了非确定的算法，True时可以自动匹配最高效率的算法
    torch.cuda.manual_seed(args.seed)    # 设置CUDA随机数
    # 日志
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)
    criterion = ConvSeparateLoss()
    criterion_val = nn.CrossEntropyLoss()   # 交叉熵
    criterion_val = criterion_val.cuda()
    criterion = criterion.cuda()    # 部署到cuda
    model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion)    #    设置model
    model = model.cuda()
    # 日志
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,     # 动量参数
        weight_decay=args.weight_decay)   # 权重衰减

    train_transform, valid_transform = utils._data_transforms_cifar10(args)      # 处理CIFAR时要使用的变量
    train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)     # 引入数据集

    num_train = len(train_data)    # 训练数据量
    indices = list(range(num_train))     # 索引数量
    split = int(np.floor(args.train_portion * num_train))      # 分割数据集和验证集

    # 训练队列设置
    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,    # 每个batch的大小
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),   # 策略采样
        pin_memory=True, num_workers=2)

    # 验证队列设置
    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True, num_workers=2)

    # 模拟退火学习率
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

    # 初始化架构
    architect = Architect(model, args)
    lr = args.learning_rate
    #开始训练
    weight1 = 1
    for epoch in range(args.epochs):
        scheduler.step()
        lr = scheduler.get_lr()[0]
        logging.info(model.arch_parameters())

        logging.info('epoch %d lr %e', epoch, lr)
        # 该算法用于选择链接边，和操作
        genotype = model.genotype()
        # 将模型架构的实际组合保存到日志
        logging.info('genotype = %s', genotype)

        logging.info(F.softmax(model.alphas_normal, dim=-1))
        logging.info(F.softmax(model.alphas_reduce, dim=-1))

        # training，acc是正确率，obj是实际对象
        train_acc, train_obj = train(train_queue, valid_queue, model, architect, criterion_val, optimizer, lr, epoch, weight1)
        logging.info('train_acc %f', train_acc)

        # validation
        valid_acc, valid_obj = infer(valid_queue, model, criterion_val)
        logging.info('valid_acc %f', valid_acc)
        # logging.info(dw)
        if train_acc < 80:
            weight1 = 10/(epoch/5+1)
        # elif 85 > train_acc >= 70:
        #     weight1 = 0
        elif train_acc >= 80:
            weight1 = -20*epoch/100
        utils.save(model, os.path.join(args.save, 'weights.pt'))

# 训练算法
def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, epoch, weight1):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    for step, (input, target) in enumerate(train_queue):
        model.train()
        n = input.size(0)
        input = Variable(input, requires_grad=False).cuda()
        target = Variable(target, requires_grad=False).cuda()     # cuda(async=True)报错

        # get a random minibatch from the search queue with replacement
        input_search, target_search = next(iter(valid_queue))     # next返回迭代器的下一项，即一个个minibatch组成
        input_search = Variable(input_search, requires_grad=False).cuda()
        target_search = Variable(target_search, requires_grad=False).cuda()

        # dw = architect.dw_exp(input, target, input_search, target_search, lr, optimizer)
        # print(dw)
        alpha_input = torch.cat([model.alphas_normal, model.alphas_reduce], dim=0)
        # 调用step中的梯度算法，其中包括前向传播，梯度计算，反向传播，此处是优化架构参数α

        loss1, loss2 = architect.step(input, target, input_search, target_search, lr, optimizer, weight1)
        # architect.dw_regularization(input, target, input_search, target_search, lr, optimizer)
        optimizer.zero_grad()     # 梯度归零
        logits = model(input)     # 计算输出
        loss = criterion(logits, target)  # 计算损失
        loss.backward()    # bp
        nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)   # 设定梯度范数阈值，防止梯度爆炸
        optimizer.step()   # 此处是优化模型参数W

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))    # 取出top1，与top5预测准确率
        objs.update(loss.item(), n)      # update是DARTS文件下utils中的函数，loss.data[0]现在改为loss.item()
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            # avg为utils中函数，用于求平均值
            # 日志信息为步数，loss，top1正确率，top5正确率
            logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
            logging.info('loss_1 %f', loss2.item())


    return top1.avg, objs.avg

# 对训练数据进行验证
def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()    # 关闭dropout和BN，此处是用于验证所以不需要dropout与BN


    for step, (input, target) in enumerate(valid_queue):
        with torch.no_grad():                   # input = Variable(input, volatile=True).cuda()
            input = Variable(input, volatile=True).cuda()
        with torch.no_grad():
            target = Variable(target, volatile=True).cuda()

        logits = model(input)
        loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)                             # n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


if __name__ == '__main__':
    main()
