import os
import time

import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import json
import numpy as np


SCALING_FACTOR = 2**10
ROOT = True

def scaling(x):
    return (int)(x * SCALING_FACTOR)

def scaling_large(x):
    return (int)(x * 2**20)

def descaling(x):
    return (int)(x // SCALING_FACTOR)

def to_val_expo(x):
    # 使用 NumPy 的 frexp 函數將浮點數分解為尾數和指數
    mantissa, exponent = np.frexp(x)
    # 將尾數轉換為整數
    val = int(mantissa * 2**10)
    # 將指數轉換為以 10 為底的指數，並限制精度為小數點後 6 位
    expo = round(exponent - 10, 6)
    return val, expo

def sigmoid_approximate_no_division(x_scaled):
    result = torch.zeros_like(x_scaled)
    result[x_scaled <= -5 * SCALING_FACTOR] = 0
    mask = (x_scaled > -5 * SCALING_FACTOR) & (x_scaled < -2.5 * SCALING_FACTOR)
    result[mask] = 25 * x_scaled[mask] + 125 * SCALING_FACTOR
    mask = (x_scaled >= -2.5 * SCALING_FACTOR) & (x_scaled <= -1 * SCALING_FACTOR)
    result[mask] = 125 * x_scaled[mask] + 375 * SCALING_FACTOR
    mask = (x_scaled > -1 * SCALING_FACTOR) & (x_scaled < 1 * SCALING_FACTOR)
    result[mask] = 250 * x_scaled[mask] + 500 * SCALING_FACTOR
    mask = (x_scaled >= 1 * SCALING_FACTOR) & (x_scaled <= 2.5 * SCALING_FACTOR)
    result[mask] = 125 * x_scaled[mask] + 625 * SCALING_FACTOR
    mask = (x_scaled > 2.5 * SCALING_FACTOR) & (x_scaled < 5 * SCALING_FACTOR)
    result[mask] = 25 * x_scaled[mask] + 875 * SCALING_FACTOR
    result[x_scaled >= 5 * SCALING_FACTOR] = SCALING_FACTOR * SCALING_FACTOR
    return result

    # if x_scaled <= -5 * SCALING_FACTOR:
    #     return 0
    # if -5 * SCALING_FACTOR < x_scaled < -2.5 * SCALING_FACTOR:
    #     return 25 * x_scaled + 125 * SCALING_FACTOR  # 將係數放大為整數
    # if -2.5 * SCALING_FACTOR <= x_scaled <= -1 * SCALING_FACTOR:
    #     return 125 * x_scaled + 375 * SCALING_FACTOR
    # if -1 * SCALING_FACTOR < x_scaled < 1 * SCALING_FACTOR:
    #     return 250 * x_scaled + 500 * SCALING_FACTOR
    # if 1 * SCALING_FACTOR <= x_scaled <= 2.5 * SCALING_FACTOR:
    #     return 125 * x_scaled + 625 * SCALING_FACTOR
    # if 2.5 * SCALING_FACTOR < x_scaled < 5 * SCALING_FACTOR:
    #     return 25 * x_scaled + 875 * SCALING_FACTOR
    # if x_scaled >= 5 * SCALING_FACTOR:
    #     return SCALING_FACTOR * SCALING_FACTOR

def sigmoid_approximate_no_division_param(x_scaled):
    if x_scaled <= -5 * SCALING_FACTOR:
        return 0, 0
    if -5 * SCALING_FACTOR < x_scaled and x_scaled < -2.5 * SCALING_FACTOR:
        return 25, 125 * SCALING_FACTOR
    if -2.5 * SCALING_FACTOR <= x_scaled and x_scaled <= -1 * SCALING_FACTOR:
        return 125, 375 * SCALING_FACTOR
    if x_scaled > -1 * SCALING_FACTOR and x_scaled < 1 * SCALING_FACTOR:
        return 250, 500 * SCALING_FACTOR
    if x_scaled >= 1 * SCALING_FACTOR and x_scaled <= 2.5 * SCALING_FACTOR:
        return 125, 625 * SCALING_FACTOR
    if x_scaled > 2.5 * SCALING_FACTOR and x_scaled < 5 * SCALING_FACTOR:
        return 25, 875 * SCALING_FACTOR
    if x_scaled >= 5 * SCALING_FACTOR:
        return 0, SCALING_FACTOR * SCALING_FACTOR


def sigmoid_approximate(x):
    result = torch.zeros_like(x)
    result[x <= -5] = 0
    mask = (x > -5) & (x < -2.5)
    result[mask] = 0.025 * x[mask] + 0.125
    mask = (x >= -2.5) & (x <= -1)
    result[mask] = 0.125 * x[mask] + 0.375
    mask = (x > -1) & (x < 1)
    result[mask] = 0.25 * x[mask] + 0.5
    mask = (x >= 1) & (x <= 2.5)
    result[mask] = 0.125 * x[mask] + 0.625
    mask = (x > 2.5) & (x < 5)
    result[mask] = 0.025 * x[mask] + 0.875
    result[x >= 5] = 1
    return result

def sigmoid_appoximate_param(x):
    if x <= -5:
        return 0, 0
    if -5 < x and x < -2.5:
        return 0.025, 0.125
    if -2.5 <= x and x <= -1:
        return 0.125, 0.375
    if x > -1 and x < 1:
        return 0.25, 0.5
    if x >= 1 and x <= 2.5:
        return 0.125, 0.625
    if x > 2.5 and x < 5:
        return 0.025, 0.875
    if x >= 5:
        return 0, 1
    
    
class InnerNode():

    def __init__(self, depth, args):
        self.args = args
        self.fc = nn.Linear(self.args.input_dim, 1)
        beta = torch.randn(1)
        #beta = beta.expand((self.args.batch_size, 1))
        if self.args.cuda:
            beta = beta.cuda()
        self.beta = nn.Parameter(beta)
        self.leaf = False
        self.prob = None
        self.leaf_accumulator = []
        self.lmbda = self.args.lmbda * 2 ** (-depth)
        self.build_child(depth)
        self.penalties = []
        self.operation_log = []
        self.left_child_id = None
        self.right_child_id = None

    def reset(self):
        self.leaf_accumulator = []
        self.penalties = []
        self.operation_log = []
        self.left.reset()
        self.right.reset()

    def build_child(self, depth):
        if depth < self.args.max_depth:
            self.left = InnerNode(depth+1, self.args)
            self.right = InnerNode(depth+1, self.args)
        else :
            self.left = LeafNode(self.args)
            self.right = LeafNode(self.args)
        self.left_child_id = id(self.left)  # 記錄左子節點的 id
        self.right_child_id = id(self.right)  # 記錄右子節點的 id

    # def forward(self, x):
    #     return(F.sigmoid(self.beta*self.fc(x)))
    
    def forward(self, x):
        linear_output = self.fc(x)
        beta_output = self.beta * linear_output
        prob = sigmoid_approximate(beta_output)  # 使用近似解
        return prob
    
    # self.prob, input_fix, fc_weight_fix, fc_bias_fix, linear_output, beta_fix, beta_output, sigmoid_a, sigmoid_b = self.forward_fix(x)
    def forward_fix(self, x):
        function_vector_scaling = np.vectorize(scaling)
        function_vector_descale = np.vectorize(descaling)

        input_fix = function_vector_scaling(x.cpu().data.numpy())
        input_fix.astype(np.int32)
        fc_weight_fix = function_vector_scaling(self.fc.weight.cpu().data.numpy())
        fc_weight_fix.astype(np.int32)
        fc_bias_fix = function_vector_scaling(self.fc.bias.cpu().data.numpy())
        fc_bias_fix.astype(np.int32)

        linear_output_temp = np.dot(input_fix, fc_weight_fix.T)
        linear_output_temp = function_vector_descale(linear_output_temp)
        linear_output_temp.astype(np.int32)
        linear_output_fix = linear_output_temp + fc_bias_fix
        
        beta_fix = function_vector_scaling(self.beta.cpu().data.numpy())
        beta_fix.astype(np.int32)
        beta_output_temp = beta_fix * linear_output_fix
        beta_output = function_vector_descale(beta_output_temp)
        beta_output.astype(np.int32)
        sigmoid_a, sigmoid_b = sigmoid_approximate_no_division_param(beta_output)
        prob_temp = beta_output * sigmoid_a
        prob = prob_temp + sigmoid_b
        prob = function_vector_descale(prob)
        prob.astype(np.int32)

        return prob, input_fix, fc_weight_fix, fc_bias_fix, linear_output_fix, beta_fix, beta_output, sigmoid_a, sigmoid_b


    def select_next(self, x):
        prob = self.forward(x)
        if prob < 0.5:
            return(self.left, prob)
        else:
            return(self.right, prob)

    def cal_prob(self, x, path_prob): #LVR過程
        self.prob = self.forward(x) #probability of selecting right node
        self.path_prob = path_prob
        left_leaf_accumulator = self.left.cal_prob(x, path_prob * (1-self.prob))
        right_leaf_accumulator = self.right.cal_prob(x, path_prob * self.prob)
        self.leaf_accumulator.extend(left_leaf_accumulator)
        self.leaf_accumulator.extend(right_leaf_accumulator)
        return(self.leaf_accumulator)
    
    def cal_prob_with_log(self, x, path_prob): #LVR過程
        global ROOT
        function_vector_scaling = np.vectorize(scaling)
        function_vector_descale = np.vectorize(descaling)

        self.prob, input_fix, fc_weight_fix, fc_bias_fix, linear_output, beta_fix, beta_output, sigmoid_a, sigmoid_b = self.forward_fix(x) #probability of selecting right node
        
        left_prob = path_prob * (SCALING_FACTOR - self.prob)
        left_prob = function_vector_descale(left_prob)
        left_prob.astype(np.int32)

        right_prob = path_prob * self.prob
        right_prob = function_vector_descale(right_prob)
        right_prob.astype(np.int32)

        self.operation_log.append({
            'node_id': id(self),  # 使用節點的 id 作為標識
            'is_root': ROOT,
            'is_leaf': False,
            'left_child_id': self.left_child_id,  # 記錄左子節點的 id
            'right_child_id': self.right_child_id,  # 記錄右子節點的 id
            'input': input_fix.tolist()[0],  # 記錄輸入的 fixc
            'fc_weight': fc_weight_fix.tolist()[0],  # 記錄 fc 的權重
            'fc_bias': fc_bias_fix.tolist()[0],  # 記錄 fc 的偏置
            'linear_output': linear_output.tolist()[0][0],  # 記錄 linear_output
            'beta': beta_fix.tolist()[0],  # 記錄 beta
            'beta_output': beta_output.tolist()[0][0],  # 記錄 beta_output
            'sigmoid_w': sigmoid_a,  # 記錄 sigmoid 的 a
            'sigmoid_b': sigmoid_b,  # 記錄 sigmoid 的 b
            'prob': self.prob.tolist()[0][0],
            'path_prob_in': path_prob.tolist()[0][0],
            'path_prob_out_left': left_prob.tolist()[0][0],
            'path_prob_out_right': right_prob.tolist()[0][0]            
        })
        ROOT = False



        
        left_leaf_accumulator = self.left.cal_prob_with_log(x, left_prob)
        right_leaf_accumulator = self.right.cal_prob_with_log(x, right_prob)
        self.leaf_accumulator.extend(left_leaf_accumulator)
        self.leaf_accumulator.extend(right_leaf_accumulator)
        return(self.leaf_accumulator)

    # def get_penalty(self):
    #     penalty = (torch.sum(self.prob * self.path_prob) / torch.sum(self.path_prob), self.lmbda)
    #     if not self.left.leaf:
    #         left_penalty = self.left.get_penalty()
    #         right_penalty = self.right.get_penalty()
    #         self.penalties.append(penalty)
    #         self.penalties.extend(left_penalty)
    #         self.penalties.extend(right_penalty)
    #     return(self.penalties)
    
    def get_penalty(self):
        # 確保 torch.sum(self.path_prob) 不為零
        path_prob_sum = torch.sum(self.path_prob)
        if path_prob_sum == 0:
            path_prob_sum = 1e-6  # 避免除以零

        penalty = (torch.sum(self.prob * self.path_prob) / path_prob_sum, self.lmbda)
        
        # 確保 penalty 值在有效範圍內
        penalty_value = penalty[0]
        if penalty_value <= 0 or penalty_value >= 1:
            penalty_value = torch.clamp(penalty_value, min=1e-6, max=1-1e-6)
            penalty = (penalty_value, self.lmbda)

        if not self.left.leaf:
            left_penalty = self.left.get_penalty()
            right_penalty = self.right.get_penalty()
            self.penalties.append(penalty)
            self.penalties.extend(left_penalty)
            self.penalties.extend(right_penalty)
        return self.penalties


class LeafNode():
    def __init__(self, args):
        self.args = args
        self.param = torch.randn(self.args.output_dim)
        if self.args.cuda:
            self.param = self.param.cuda()
        self.param = nn.Parameter(self.param)
        self.leaf = True
        self.softmax = nn.Softmax()

    def forward(self):
        return(self.softmax(self.param.view(1,-1)))

    def reset(self):
        pass

    def cal_prob(self, x, path_prob):
        Q = self.forward()
        #Q = Q.expand((self.args.batch_size, self.args.output_dim))
        Q = Q.expand((path_prob.size()[0], self.args.output_dim))
        return([[path_prob, Q]])

    def cal_prob_with_log(self, x, path_prob):
        Q = self.forward()
        # Q = Q.expand((path_prob.size()[0], self.args.output_dim))
        function_vector_scaling = np.vectorize(scaling)
        Q_output = function_vector_scaling(Q.cpu().data.numpy())
        Q_output.astype(np.int32)

        self.operation_log.append({
            'node_id': id(self),  # 使用節點的 id 作為標識
            'is_leaf': True,
            'is_final_output': False,
            'path_prob': path_prob.tolist()[0][0],
            # 'Q': Q.cpu().data.numpy().tolist()
            'Q': Q_output.tolist()[0]
        })
        return [[path_prob, Q_output]]

class SoftDecisionTree(nn.Module):

    def __init__(self, args):
        super(SoftDecisionTree, self).__init__()
        self.args = args
        self.root = InnerNode(1, self.args)
        self.collect_parameters() ##collect parameters and modules under root node
        self.optimizer = optim.SGD(self.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        self.test_acc = []
        self.define_extras(self.args.batch_size)
        self.best_accuracy = 0.0

    def define_extras(self, batch_size):
        ##define target_onehot and path_prob_init batch size, because these need to be defined according to batch size, which can be differ
        self.target_onehot = torch.FloatTensor(batch_size, self.args.output_dim)
        self.target_onehot = Variable(self.target_onehot)
        self.path_prob_init = Variable(torch.ones(batch_size, 1))
        if self.args.cuda:
            self.target_onehot = self.target_onehot.cuda()
            self.path_prob_init = self.path_prob_init.cuda()
    
    def forward(self, x):
        
        self.path_prob_init = self.path_prob_init.cpu().data.numpy()
        function_vector_scaling = np.vectorize(scaling)
        self.path_prob_init = function_vector_scaling(self.path_prob_init)
        self.path_prob_init.astype(np.int32)
        leaf_accumulator = self.root.cal_prob_with_log(x, self.path_prob_init)
        # leaf_accumulator = self.root.cal_prob(x, self.path_prob_init)
        batch_size = x.size(0)
        max_prob = [-1. for _ in range(batch_size)]
        max_Q = [None for _ in range(batch_size)]
        
        for path_prob, Q in leaf_accumulator:
            path_prob_numpy = path_prob.reshape(-1)
            for i in range(batch_size):
                if max_prob[i] < path_prob_numpy[i]:
                    max_prob[i] = path_prob_numpy[i]
                    max_Q[i] = Q[i]

        max_Q = [torch.tensor(q) if isinstance(q, np.ndarray) else q for q in max_Q]        
        output = torch.stack(max_Q)
        return output

    def cal_loss(self, x, y):           # x: ML input data(graph), y: target
        batch_size = y.size()[0]
        leaf_accumulator = self.root.cal_prob(x, self.path_prob_init)
        loss = 0.
        max_prob = [-1. for _ in range(batch_size)]
        max_Q = [torch.zeros(self.args.output_dim) for _ in range(batch_size)]
        for (path_prob, Q) in leaf_accumulator:
            TQ = torch.bmm(y.view(batch_size, 1, self.args.output_dim), torch.log(Q).view(batch_size, self.args.output_dim, 1)).view(-1,1)
            loss += path_prob * TQ
            path_prob_numpy = path_prob.cpu().data.numpy().reshape(-1)
            for i in range(batch_size):
                if max_prob[i] < path_prob_numpy[i]:
                    max_prob[i] = path_prob_numpy[i]
                    max_Q[i] = Q[i]
        loss = loss.mean()
        penalties = self.root.get_penalty()
        C = 0.
        for (penalty, lmbda) in penalties:
            C -= lmbda * 0.5 *(torch.log(penalty) + torch.log(1-penalty))
        output = torch.stack(max_Q)
        self.root.reset() ##reset all stacked calculation
        return(-loss + C, output) ## -log(loss) will always output non, because loss is always below zero. I suspect this is the mistake of the paper?


    def cal_kd_loss(self, x, teacher_model, target, alpha=0.5, temperature=3.0):
        batch_size = target.size()[0]
        leaf_accumulator = self.root.cal_prob(x, self.path_prob_init)
        loss = 0.
        max_prob = [-1. for _ in range(batch_size)]
        max_Q = [torch.zeros(self.args.output_dim) for _ in range(batch_size)]

        all_leaf_outputs = []
        for (path_prob, Q) in leaf_accumulator:
            TQ = torch.bmm(target.view(batch_size, 1, self.args.output_dim), torch.log(Q).view(batch_size, self.args.output_dim, 1)).view(-1,1)
            loss += path_prob * TQ
            path_prob_numpy = path_prob.cpu().data.numpy().reshape(-1)
            for i in range(batch_size):
                if max_prob[i] < path_prob_numpy[i]:
                    max_prob[i] = path_prob_numpy[i]
                    max_Q[i] = Q[i]
            all_leaf_outputs.append(Q)

        loss = loss.mean()
        penalties = self.root.get_penalty()
        C = 0.
        for (penalty, lmbda) in penalties:
            if penalty <= 0 or penalty >= 1:
                penalty = torch.clamp(penalty, min=1e-6, max=1-1e-6)
            C -= lmbda * 0.5 *(torch.log(penalty) + torch.log(1-penalty))
        output = torch.stack(max_Q)
        self.root.reset() ##reset all stacked calculation
        
        with torch.no_grad():
            teacher_preds = teacher_model(x)
        # 知識蒸餾損失
        kd_loss = F.kl_div(F.log_softmax(torch.stack(all_leaf_outputs) / temperature, dim=1),
                       F.softmax(teacher_preds / temperature, dim=1),
                       reduction='batchmean') * (temperature ** 2)
        
        # 標準交叉熵損失
        ce_loss = -loss + C
        
        # 總損失
        total_loss = alpha * kd_loss + (1.0 - alpha) * ce_loss
        return total_loss, output
    
    def collect_parameters(self):
        nodes = [self.root]
        self.module_list = nn.ModuleList()
        self.param_list = nn.ParameterList()
        while nodes:
            node = nodes.pop(0)
            if node.leaf:
                param = node.param
                self.param_list.append(param)
            else:
                fc = node.fc
                beta = node.beta
                nodes.append(node.right)
                nodes.append(node.left)
                self.param_list.append(beta)
                self.module_list.append(fc)

    def train_(self, train_loader, epoch):
        self.train()
        self.define_extras(self.args.batch_size)
        for batch_idx, (data, target) in enumerate(train_loader):
            correct = 0
            if self.args.cuda:
                data, target = data.cuda(), target.cuda()
            #data = data.view(self.args.batch_size,-1)
            target = Variable(target)
            target_ = target.view(-1,1)
            batch_size = target_.size()[0]
            data = data.view(batch_size,-1)
            ##convert int target to one-hot vector
            data = Variable(data)
            if not batch_size == self.args.batch_size: #because we have to initialize parameters for batch_size, tensor not matches with batch size cannot be trained
                self.define_extras(batch_size)
            self.target_onehot.data.zero_()            
            self.target_onehot.scatter_(1, target_, 1.)
            self.optimizer.zero_grad()

            loss, output = self.cal_loss(data, self.target_onehot)
            #loss.backward(retain_variables=True)
            loss.backward()
            self.optimizer.step()
            pred = output.data.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(target.data).cpu().sum()
            accuracy = 100. * correct / len(data)

            if batch_idx % self.args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {}/{} ({:.4f}%)'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item(),
                    correct, len(data),
                    accuracy))
                
    def train_kd_(self, train_loader, epoch, teacher_model):
        self.train()
        self.define_extras(self.args.batch_size)
        for batch_idx, (data, target) in enumerate(train_loader):
            correct = 0
            if self.args.cuda:
                data, target = data.cuda(), target.cuda()
            #data = data.view(self.args.batch_size,-1)
            target = Variable(target)
            target_ = target.view(-1,1)
            batch_size = target_.size()[0]
            data = data.view(batch_size,-1)
            ##convert int target to one-hot vector
            data = Variable(data)
            if not batch_size == self.args.batch_size: #because we have to initialize parameters for batch_size, tensor not matches with batch size cannot be trained
                self.define_extras(batch_size)
            self.target_onehot.data.zero_()            
            self.target_onehot.scatter_(1, target_, 1.)
            self.optimizer.zero_grad()

            loss, output = self.cal_kd_loss(data, teacher_model, self.target_onehot)
            #loss.backward(retain_variables=True)
            loss.backward()
            self.optimizer.step()
            pred = output.data.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(target.data).cpu().sum()
            accuracy = 100. * correct / len(data)

            if batch_idx % self.args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {}/{} ({:.4f}%)'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item(),
                    correct, len(data),
                    accuracy))
                
                
    def train_kd_one_time_upgrade_(self, train_loader, teacher_model):
        self.train()
        self.define_extras(self.args.batch_size)
        for data, target in train_loader:
            if self.args.cuda:
                data, target = data.cuda(), target.cuda()
            target = Variable(target)
            target_ = target.view(-1,1)
            batch_size = target_.size()[0]
            data = data.view(batch_size,-1)
            data = Variable(data)
            if not batch_size == self.args.batch_size: #because we have to initialize parameters for batch_size, tensor not matches with batch size cannot be trained
                self.define_extras(batch_size)
            self.target_onehot = torch.zeros(batch_size, self.args.output_dim).to(data.device)  # 確保大小正確
            self.target_onehot.scatter_(1, target_, 1.)
            self.optimizer.zero_grad()

            # student_loss, output = self.cal_loss(data, self.target_onehot)
            # print(student_loss)

            kd_loss, _ = self.cal_kd_loss(data, teacher_model, self.target_onehot)
            # print(kd_loss)
            # teacher_loss = 
            kd_loss.backward()
            # self.optimizer.step()
            leanrning_rate = self.args.lr
            
            function_vector_scaling = np.vectorize(scaling_large)
            gradient_log = []
            with torch.no_grad():
                for param in self.parameters():
                    # 儲存原先的param
                    param_original = param.clone()
                    fix_param_original = function_vector_scaling(param_original.cpu().data.numpy())
                    fix_param_original.astype(np.int32)
                    # 取得梯度
                    descent = leanrning_rate * param.grad
                    fix_descent = function_vector_scaling(descent.cpu().data.numpy())
                    fix_descent.astype(np.int32)
                    
                    fix_param_updated = fix_param_original - fix_descent
                    
                    param -= leanrning_rate * param.grad
                    
                        
                    gradient_log.append({
                        'param_original': fix_param_original.flatten().tolist(),
                        'descent': fix_descent.flatten().tolist(),
                        'param_updated': fix_param_updated.flatten().tolist()
                    })
            
            json.dump(gradient_log, open('./extraction_gradient_descent/student_gradient_descent.json', 'w'))
                    


    def test_(self, test_loader, epoch):
        self.eval()
        self.define_extras(self.args.batch_size)
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            if self.args.cuda:
                data, target = data.cuda(), target.cuda()
            target = Variable(target)
            target_ = target.view(-1,1)
            batch_size = target_.size()[0]
            data = data.view(batch_size,-1)
            ##convert int target to one-hot vector
            data = Variable(data)
            if not batch_size == self.args.batch_size: #because we have to initialize parameters for batch_size, tensor not matches with batch size cannot be trained
                self.define_extras(batch_size)
            self.target_onehot.data.zero_()            
            self.target_onehot.scatter_(1, target_, 1.)
            _, output = self.cal_loss(data, self.target_onehot)
            pred = output.data.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(target.data).sum().item()
        accuracy = 100. * correct / len(test_loader.dataset)
        print('\nTest set: Accuracy: {}/{} ({:.4f}%)\n'.format(
            correct, len(test_loader.dataset),
            accuracy))
        self.test_acc.append(accuracy)

        if accuracy > self.best_accuracy:
            self.save_best('./result')
            self.best_accuracy = accuracy

    def test_one_graph(self, test_loader, epoch):
        self.eval()
        self.define_extras(self.args.batch_size)
        with torch.no_grad():
            for data, target in test_loader:
                if self.args.cuda:
                    data, target = data.cuda(), target.cuda()
                target = Variable(target)
                target_ = target.view(-1,1)
                batch_size = target_.size()[0]
                data = data.view(batch_size,-1)
                ##convert int target to one-hot vector
                data = Variable(data)
                if not batch_size == self.args.batch_size: #because we have to initialize parameters for batch_size, tensor not matches with batch size cannot be trained
                    self.define_extras(batch_size)
                self.target_onehot.data.zero_()            
                self.target_onehot.scatter_(1, target_, 1.)
                

                output = self.forward(data)
                pred = output.data.max(1)[1] # get the index of the max log-probability

                output_matrix = output.data.cpu().numpy()
                print("Model output matrix:\n", output_matrix)
                
                # 顯示圖片
                img = data.view(28, 28).cpu().numpy()
                plt.imshow(img, cmap='gray')
                plt.title(f'Predicted: {pred.item()}')
                plt.show()

                break
        self.save_operation_log('./extraction_student/student_extraction.json', output_matrix)

    def save_best(self, path):
        try:
            os.makedirs('./result')
        except:
            print('directory ./result already exists')

        with open(os.path.join(path, 'best_model.pkl'), 'wb') as output_file:
            pickle.dump(self, output_file)

    def test_forward(self, test_loader, epoch):
        self.eval()
        self.define_extras(self.args.batch_size)
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            if self.args.cuda:
                data, target = data.cuda(), target.cuda()
            target = Variable(target)
            target_ = target.view(-1,1)
            batch_size = target_.size()[0]
            data = data.view(batch_size,-1)
            ##convert int target to one-hot vector
            data = Variable(data)
            if not batch_size == self.args.batch_size: #because we have to initialize parameters for batch_size, tensor not matches with batch size cannot be trained
                self.define_extras(batch_size)
            self.target_onehot.data.zero_()            
            self.target_onehot.scatter_(1, target_, 1.)
            output = self.forward(data)
            pred = output.data.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(target.data).sum().item()

        accuracy = 100. * correct / len(test_loader.dataset)
        print('\nTest set: Accuracy: {}/{} ({:.4f}%)\n'.format(
            correct, len(test_loader.dataset),
            accuracy))
        self.test_acc.append(accuracy)


    def treiverse_and_extract_to_json(self, test_loader):
        self.eval()
        self.define_extras(self.args.batch_size)
        with torch.no_grad():
            for data, target in test_loader:
                if self.args.cuda:
                    data, target = data.cuda(), target.cuda()
                target = Variable(target)
                target_ = target.view(-1,1)
                batch_size = target_.size()[0]
                data = data.view(batch_size,-1)
                ##convert int target to one-hot vector
                data = Variable(data)
                if not batch_size == self.args.batch_size: #because we have to initialize parameters for batch_size, tensor not matches with batch size cannot be trained
                    self.define_extras(batch_size)
                self.target_onehot.data.zero_()            
                self.target_onehot.scatter_(1, target_, 1.)
                
                output0 = self.forward(data)
                output0_matrix = output0.data.cpu().numpy()
                print("Model output0 matrix:\n", output0_matrix)

                break
        
        self.save_operation_log('./extraction_student/operation_log.json')

    def get_operation_log(self, output_matrix):
        logs = []
        nodes = [self.root]
        while nodes:
            node = nodes.pop(0)
            # 找 Q = output_matrix 的節點，如果找到則將 is_final_output 設為 True
            if node.leaf:
                if np.array_equal(node.operation_log[0]['Q'], output_matrix[0]):
                    node.operation_log[0]['is_final_output'] = True
            logs.extend(node.operation_log)
            if not node.leaf:
                nodes.append(node.left)
                nodes.append(node.right)
        return logs


    def save_operation_log(self, filename, output_matrix):
        with open(filename, 'w') as f:
            json.dump(self.get_operation_log(output_matrix), f, indent=4)






class TeacherModel(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(TeacherModel, self).__init__()
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.fc1 = nn.Linear(784, 1200)
        self.fc2 = nn.Linear(1200, 1200)
        self.fc3 = nn.Linear(1200, num_classes)
        self.dropout = nn.Dropout(p=0.5)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu1(x)

        x = self.fc2(x)
        x = self.dropout(x)
        x = self.relu2(x)

        x = self.fc3(x)
        # x = self.softmax(x)

        return x