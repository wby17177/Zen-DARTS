import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from torch.testing._internal.common_nn import multimarginloss_1d_input_0d_target_no_reduce_test

import torch.nn.functional as F
class ConvSeparateLoss(nn.modules.loss._Loss):
    def __init__(self, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean'):
        super(ConvSeparateLoss, self).__init__(size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.weight = 1

    def forward(self, input1, target1, input2, weight1):
        input2 = F.softmax(input2, dim=-1)
        alpha_exp = torch.mean(input2, dim=1)
        exp_tensor_1 = [[i.tolist() for _ in range(input2.size(1))] for i in alpha_exp]
        loss = torch.nn.functional.cross_entropy(input1, target1)
        loss_1 = torch.nn.functional.mse_loss(input2, torch.tensor(exp_tensor_1, requires_grad=False).cuda())
        # print(loss.item())
        if abs(self.weight*weight1*loss_1.item()) > loss.item():
            self.weight = self.weight/10

        return torch.nn.functional.cross_entropy(input1, target1) + loss_1 * weight1*self.weight, loss_1 * weight1*self.weight



def _concat(xs):
    # 按行拼接x.view(-1)，view(-1)为拉成1行
    return torch.cat([x.view(-1) for x in xs])


class Architect(object):

    def __init__(self, model, args):
        self.network_momentum = args.momentum
        self.network_weight_decay = args.weight_decay
        self.model = model
        # 定义优化器

        self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
                                          lr=args.arch_learning_rate, betas=(0.5, 0.999),
                                          weight_decay=args.arch_weight_decay)

    # 该函数的作用是更新架构参数，并加载到模型中
    def _construct_model_from_theta(self, theta):
        # 新建对应参数及参数对应的索引
        model_new = self.model.new()
        model_dict = self.model.state_dict()
        params, offset = {}, 0
        # model.named_parameters()       # 给出了网络层的名字和参数

        for k, v in self.model.named_parameters():    # k，v分别为网络层名，和对应的参数tensor
            v_length = np.prod(v.size())              # 将参数tensor所有属性乘起来，得到元素个数
            params[k] = theta[offset: offset + v_length].view(v.size())     # 将网络名作为索引和对应的参数建立起对应
            offset += v_length

        assert offset == len(theta)      # 判定是否取完参数
        model_dict.update(params)        # 用params中的参数对model_dict中的参数进行覆盖
        model_new.load_state_dict(model_dict)     # 将参数加载到模型中，需要传入一个含有网络名和参数的字典
        return model_new.cuda()


    # 计算混合微分的展开近似值
    def _hessian_vector_product(self, vector, input, target, r=1e-2):
        # 得到拉格朗日函数的极小值，ξ = r / norm(dwLval(w',α))
        R = r / _concat(vector).norm()

        # W+ = W + ▽w'Lval(w',α)*ξ
        for p, v in zip(self.model.parameters(), vector):      # zip将两个list元素一一对应起来，形成一个新的list
            p.data.add_(R, v)         # p = p + R*v

        loss = self.model._loss(input, target)     #loss计算
        grads_p = torch.autograd.grad(loss, self.model.arch_parameters())     #梯度计算▽αLtrain(w',α)即架构梯度

        # W- = W+ - 2*▽w'Lav(w',α)*ξ = W - ▽w'Lval(w',α)*ξ
        for p, v in zip(self.model.parameters(), vector):
            p.data.sub_(2 * R, v)
        loss = self.model._loss(input, target)
        grads_n = torch.autograd.grad(loss, self.model.arch_parameters())

        # W = W- + ▽w'Lval(w',α)*ξ 回调到最初的权重W
        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)

        # 返回▽^2(α，w)Ltrain(w,α) * ▽w'Ltrain(w',α)的拉格朗日近似，即(W+ - W-) / 2ξ
        return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]


    # 计算未展开模型的权重
    def _compute_unrolled_model(self, input, target, eta, network_optimizer):
        input2 = torch.cat([self.model.alphas_normal, self.model.alphas_reduce], dim=0)
        loss = self.model._loss(input, target, input2)     # 计算loss
        theta = _concat(self.model.parameters()).data     # 拼接得到模型参数
        try:
            # 取出动量参数，并与动量做乘积，在进行拼接
            moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(
                self.network_momentum)
        except:
            moment = torch.zeros_like(theta)
        # gt + weight_decay * w = gt+1
        dtheta = _concat(torch.autograd.grad(loss, self.model.parameters())).data + self.network_weight_decay * theta
        # wt+1 = wt - Ir*(α*v + ε*gt+1)
        unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment + dtheta))
        return unrolled_model


    def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer):
        unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)
        unrolled_loss = unrolled_model._loss(input_valid, target_valid)

        unrolled_loss.backward()
        dalpha = [v.grad for v in unrolled_model.arch_parameters()]    # ▽αLval(w',α)
        vector = [v.grad.data for v in unrolled_model.parameters()]    # Lw'train(w',α)
        implicit_grads = self._hessian_vector_product(vector, input_train, target_train)

        # 用近似法计算 ▽αLval(w',α) - ε▽^2(α，w)Ltrain(w,α) * ▽w'Ltrain(w',α)
        for g, ig in zip(dalpha, implicit_grads):
            g.data.sub_(eta, ig.data)

        # 跟新梯度tensor
        for v, g in zip(self.model.arch_parameters(), dalpha):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)

    # cross = ConvSeparateLoss()
    # 该函数选择调用展开或非展开的跟新方式
    def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, weight1):
        # self.optimizer.zero_grad()
        # if unrolled:
        #     self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, eta, network_optimizer)
        # else:
        #     self._backward_step(input_train, target_train, input_valid, target_valid, eta, network_optimizer)
        # self.optimizer.step()
        self.optimizer.zero_grad()
        input2 = torch.cat([self.model.alphas_normal, self.model.alphas_reduce], dim=0)
        loss, loss2 = self.model._loss(input_valid, target_valid, input2, weight1)
        loss.backward()
        self.optimizer.step()
        return loss, loss2

    # 反向传播
    # def _backward_step(self, input_valid, target_valid):
    #     loss = self.model._loss(input_valid, target_valid)
    #     loss.backward()

    def _backward_step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer):
        input2 = torch.cat([self.model.alphas_normal, self.model.alphas_reduce], dim=0)
        loss = self.alpha_loss(input_valid, target_valid, input2)
        loss.backward()

    def dw_exp(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer):
        unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)
        input2 = torch.cat([self.model.alphas_normal, self.model.alphas_reduce], dim=0)
        unrolled_loss = unrolled_model._loss(input_valid, target_valid, input2)

        unrolled_loss.backward()
        vector = [v.grad.data for v in unrolled_model.parameters()]
        dw_exp = []
        for i in vector:
            dw_exp.append(torch.mean(i))
        return dw_exp

    def dw_regularization(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer):
        unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)
        input2 = torch.cat([self.model.alphas_normal, self.model.alphas_reduce], dim=0)
        unrolled_loss = unrolled_model._loss(input_valid, target_valid, input2)
        unrolled_loss.backward()
        vector = [v.grad.data for v in unrolled_model.parameters()]

        self.optimizer.zero_grad()
        for v, g in zip(self.model.arch_parameters(), vector):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)
        self.optimizer.zero_grad()
        self.optimizer.step()


    def alpha_loss(self, input_val, target_val, input2):
        alpha_exp = torch.mean(input2)

        loss_1 = torch.nn.functional.mse_loss(input2, torch.tensor(alpha_exp.item(), requires_grad=False).cuda())
        return torch.nn.functional.cross_entropy(input_val, target_val)-0.05*loss_1


    # 正则化限制架构参数
    # def mlc_loss(self, arch_param):
    #     y_pred_neg = arch_param
    #     neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    #     aux_loss = torch.mean(neg_loss)
    #     return aux_loss
    #
    # def _backward_step(self, input_valid, target_valid, epoch):
    #     weights = 0 + 50 * epoch / 100
    #     ssr_normal = self.mlc_loss(self.model._arch_parameters)
    #     loss = self.model._loss(self.input_valid, target_valid) + weights * ssr_normal
    #     # loss = self._val_loss(self.model, input_valid, target_valid)
    #     loss.backward()
