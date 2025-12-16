import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *  # 使用*引入，可以直接使用该文件中所有的函数
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype
from entmax import entmax15

# 混合操作
class MixedOp(nn.Module):

    def __init__(self, C, stride):
        super(MixedOp, self).__init__()
        # 定义操作空间，ModuleList是存放网络结构的List
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, False)   # 从字典OPS中取得对应操作并进行计算
            if 'pool' in primitive:     # 判定当前索引是否为pool，取到pool操作则在后面紧跟BatchNorm层
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)

    # 每个操作乘上对应的参数w，求和形成混合操作的前向传播
    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))


# 构建Cell
class Cell(nn.Module):

    # C_prev_prev是上上一个输入的通道，C_prev是上一个输入的通道
    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        self.reduction = reduction
        # 判断是否是上一个reduction cell
        # 由于上一个Cell为reduction cell时，上上一个Cell的跳跃输出与当前Cell的输入尺寸(h,w)不匹配，所以需要进行一个Reduce操作
        if reduction_prev:
            # preprocess0为当前Cell的input0, preprocess1为当前Cell的input1
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps    # 每个Cell中待搜索的节点
        self._multiplier = multiplier    # 当前已搜索节点

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()
        for i in range(self._steps):
            # 2代表Cell中两个原始输入节点
            for j in range(2 + i):
                # reduction是对原始输入的两个节点进行尺寸缩减
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride)
                # _ops中存储的是每个节点的混合操作
                self._ops.append(op)

    def forward(self, s0, s1, weights):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        # h为输入，weights为权重参数,j为操作长度
        for i in range(self._steps):
            s = sum(self._ops[offset + j](h, weights[offset + j]) for j, h in enumerate(states))
            offset += len(states)     # offset记录上一个节点的最后一个输入的位置，即下一个节点的起始位置。由于weight，和_ops是顺序存放所有的操做的输入和权重。
            states.append(s)     # 当前节点的输出加入下一个节点的输入列表

        return torch.cat(states[-self._multiplier:], dim=1)


# 构建supernet
class Network(nn.Module):

    def __init__(self, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3):
        super(Network, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion    # 损失计算
        self._steps = steps
        self._multiplier = multiplier

        C_curr = stem_multiplier * C      # 扩大通道数
        # 对头部进行处理
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),     # 原始输入为rgb三通道图像
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:   # 在三分之一处和三分之二处使用reduction cell
                C_curr *= 2        # 通道数扩大，扩大特征图的数量，以提取更多特征
                reduction = True
            else:
                reduction = False
            # 虽然Cell的输入尺寸和通道可能不同，但是结构相同
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            # 由于Cell中每个节点都可能作为输出，所以supernet中，当前输入的维度会扩大
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)   # 构建平均pooling，size为1*1
        self.classifier = nn.Linear(C_prev, num_classes)     # 线性层，用于分类

        self._initialize_alphas()     # 初始化参数
    # 新建一个Network,并将其及其参数复制到指定的目录
    def new(self):
        model_new = Network(self._C, self._num_classes, self._layers, self._criterion).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    # 前向传播
    def forward(self, input):

        s0 = s1 = self.stem(input)      # 处理头部
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = F.softmax(self.alphas_reduce, dim=-1)     # 用softmax对cell进行赋值
            else:
                weights = F.softmax(self.alphas_normal, dim=-1)
            # 迭代传播Cell
            s0, s1 = s1, cell(s0, s1, weights)
        out = self.global_pooling(s1)      # s1中存储Cell的最终输出
        logits = self.classifier(out.view(out.size(0), -1))        # 经过线性分类层的最终结果
        return logits

    # 计算loss
    def _loss(self, input, target, input2, weight1):
        logits = self(input)
        return self._criterion(logits, target, input2, weight1)

    # 初始化参数
    def _initialize_alphas(self):
        # 计算边数总数，_steps为中间节点的数量
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(PRIMITIVES)    # 操作的数量

        # α_normal和α_reduce存储的是架构参数tensor，行为边数，列为操作数，行列表示某条边选择某个操作
        self.alphas_normal = Variable(8*torch.ones(k, num_ops).cuda()-1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True)
        self.alphas_reduce = Variable(4*torch.ones(k, num_ops).cuda()-1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True)
        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
        ]

    def arch_parameters(self):
        return self._arch_parameters

    def genotype(self):

        def _parse(weights):
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n   # 某个节点的链接边数量
                W = weights[start:end].copy()  # 取出并复制当前节点的所有链接边
                # 搜索策略：边搜索，保留具有最大混合权重的两条链接边。
                edges = sorted(range(i + 2),
                               key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[
                        :2]
                # 对于链接边，再搜索最大权重的操作
                for j in edges:
                    k_best = None
                    k_sec = None
                    for k in range(len(W[j])):
                        if k != PRIMITIVES.index('none'):
                            if k_sec is None:
                                k_sec=k
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_sec = k_best
                                k_best = k
                    # 保存对应链接边选择的操作
                    # if W[j][k_best] <= 1.2*W[j][k_sec] and k_best == PRIMITIVES.index('skip_connect'):
                    #     gene.append((PRIMITIVES[k_sec], j))
                    #     continue
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene

        gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
        gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

        # self._steps是待选节点总数，multiplier是当前已搜索节点数，
        concat = range(2 + self._steps - self._multiplier, self._steps + 2)     # 计算被链接的中间节点的序号
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
        )
        return genotype