import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import json

SCALING_FACTOR = 2**10
def scaling(x):
    return (int)(x * SCALING_FACTOR)

def descaling(x):
    return (int)(x // SCALING_FACTOR)




class TeacherModel(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(TeacherModel, self).__init__()
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.fc1 = nn.Linear(784, 1200)
        self.fc2 = nn.Linear(1200, 1200)
        self.fc3 = nn.Linear(1200, num_classes)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu1(x)

        x = self.fc2(x)
        x = self.dropout(x)
        x = self.relu2(x)

        x = self.fc3(x)

        return x

# 創建資料夾
extraction_teacher_filename = 'extraction_teacher_whole'
if not os.path.exists(f'{extraction_teacher_filename}'):
    os.makedirs(f'{extraction_teacher_filename}')

for layer in ['fc1', 'fc2', 'fc3', 'relu1', 'relu2']:
    if not os.path.exists(f'{extraction_teacher_filename}/{layer}'):
        os.makedirs(f'{extraction_teacher_filename}/{layer}')

def pickle_store(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def hook_fc1(model, input, output):
    function_scaling = np.vectorize(scaling)
    function_descaling = np.vectorize(descaling)
    layer_input = input[0].detach().cpu().numpy()
    layer_output = output.detach().cpu().numpy()
    layer_weight = model.weight.data.detach().cpu().numpy()
    layer_bias = model.bias.data.detach().cpu().numpy()
    # 隨機選擇 0-1200 其中一個 neuron 當作checkpoint
    
    int_layer_weight = function_scaling(layer_weight)
    int_layer_bias = function_scaling(layer_bias)
    int_layer_input = function_scaling(layer_input)
    
    int_layer_weight.astype(np.int32)
    int_layer_bias.astype(np.int32)
    int_layer_input.astype(np.int32)
    
    linear_output_temp = np.dot(int_layer_input, int_layer_weight.T)
    linear_output_temp = function_descaling(linear_output_temp)
    linear_output_temp.astype(np.int32)
    linear_output = linear_output_temp + int_layer_bias
    
    print('fc1')
    print('input:', input[0].shape)
    print('linear_output:', linear_output.shape)
    print('layer_weight:', int_layer_weight.shape)
    print('layer_bias:', int_layer_bias.shape)
    
    # 將input, output, checkpoint_weight, checkpoint_bias, calculated_output 存成json檔案
    operation_log = {
            'layer': 'fc1',
            'input': int_layer_input.tolist()[0],
            'linear_output': linear_output.tolist()[0],
            'layer_weight': int_layer_weight.tolist(),
            'layer_bias': int_layer_bias.tolist()
            
        }
    with open(f'{extraction_teacher_filename}/fc1/fc1_operation_log.json', 'w') as f:
        json.dump(operation_log, f)
        
        
def hook_fc2(model, input, output):
    function_scaling = np.vectorize(scaling)
    function_descaling = np.vectorize(descaling)
    layer_input = input[0].detach().cpu().numpy()
    layer_output = output.detach().cpu().numpy()
    layer_weight = model.weight.data.detach().cpu().numpy()
    layer_bias = model.bias.data.detach().cpu().numpy()
    # 隨機選擇 0-1200 其中一個 neuron 當作checkpoint
    
    int_layer_weight = function_scaling(layer_weight)
    int_layer_bias = function_scaling(layer_bias)
    int_layer_input = function_scaling(layer_input)
    
    int_layer_weight.astype(np.int32)
    int_layer_bias.astype(np.int32)
    int_layer_input.astype(np.int32)
    
    linear_output_temp = np.dot(int_layer_input, int_layer_weight.T)
    linear_output_temp = function_descaling(linear_output_temp)
    linear_output_temp.astype(np.int32)
    linear_output = linear_output_temp + int_layer_bias
    
    print('fc2')
    print('input:', input[0].shape)
    print('linear_output:', linear_output.shape)
    print('layer_weight:', int_layer_weight.shape)
    print('layer_bias:', int_layer_bias.shape)
    
    # 將input, output, checkpoint_weight, checkpoint_bias, calculated_output 存成json檔案
    operation_log = {
            'layer': 'fc2',
            'input': int_layer_input.tolist()[0],
            'linear_output': linear_output.tolist()[0],
            'layer_weight': int_layer_weight.tolist(),
            'layer_bias': int_layer_bias.tolist()
            
        }
    with open(f'{extraction_teacher_filename}/fc2/fc2_operation_log.json', 'w') as f:
        json.dump(operation_log, f)
        
        
def hook_fc3(model, input, output):
    function_scaling = np.vectorize(scaling)
    function_descaling = np.vectorize(descaling)
    layer_input = input[0].detach().cpu().numpy()
    layer_output = output.detach().cpu().numpy()
    layer_weight = model.weight.data.detach().cpu().numpy()
    layer_bias = model.bias.data.detach().cpu().numpy()
    # 隨機選擇 0-1200 其中一個 neuron 當作checkpoint
    
    int_layer_weight = function_scaling(layer_weight)
    int_layer_bias = function_scaling(layer_bias)
    int_layer_input = function_scaling(layer_input)
    
    int_layer_weight.astype(np.int32)
    int_layer_bias.astype(np.int32)
    int_layer_input.astype(np.int32)
    
    linear_output_temp = np.dot(int_layer_input, int_layer_weight.T)
    linear_output_temp = function_descaling(linear_output_temp)
    linear_output_temp.astype(np.int32)
    linear_output = linear_output_temp + int_layer_bias
    
    print('fc3')
    print('input:', input[0].shape)
    print('linear_output:', linear_output.shape)
    print('layer_weight:', int_layer_weight.shape)
    print('layer_bias:', int_layer_bias.shape)
    
    # 將input, output, checkpoint_weight, checkpoint_bias, calculated_output 存成json檔案
    operation_log = {
            'layer': 'fc3',
            'input': int_layer_input.tolist()[0],
            'linear_output': linear_output.tolist()[0],
            'layer_weight': int_layer_weight.tolist(),
            'layer_bias': int_layer_bias.tolist()
            
        }
    with open(f'{extraction_teacher_filename}/fc3/fc3_operation_log.json', 'w') as f:
        json.dump(operation_log, f)
    



    
    
def hook_relu(module, input, output, layer_name):
    relu_input = input[0].detach().cpu().numpy()
    relu_output = output.detach().cpu().numpy()
    function_scaling = np.vectorize(scaling)


    function_reluoperation = np.vectorize(relu_operation)
    relu_operator = function_reluoperation(relu_input)
    relu_input_nume = function_scaling(relu_input)
    relu_output_nume = function_scaling(relu_output)

    relu_input_nume.astype(np.int32)
    relu_output_nume.astype(np.int32)

    # 將input, output, relu_operator 存成json檔案
    operation_log = {
            'layer': layer_name,
            'input': relu_input_nume.tolist()[0],
            'output': relu_output_nume.tolist()[0],
            'relu_operator': relu_operator.tolist()[0]
        }
    
    with open(f'{extraction_teacher_filename}/{layer_name}/{layer_name}_operation_log.json', 'w') as f:
        json.dump(operation_log, f)



def relu_operation(x):
    if x < 0:
        x = 0
    else:
        x = 1
    return (x)

if __name__ == '__main__':
    # 直接載入整個模型
    teachermodel = torch.load('teacher.pkl')

    # 設置 CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teachermodel.to(device)

    # 註冊 hook
    teachermodel.fc1.register_forward_hook(lambda module, input, output: hook_fc1(module, input, output))
    teachermodel.fc2.register_forward_hook(lambda module, input, output: hook_fc2(module, input, output))
    teachermodel.fc3.register_forward_hook(lambda module, input, output: hook_fc3(module, input, output))
    teachermodel.relu1.register_forward_hook(lambda module, input, output: hook_relu(module, input, output, 'relu1'))
    teachermodel.relu2.register_forward_hook(lambda module, input, output: hook_relu(module, input, output, 'relu2'))

    # 加載測試數據
    test_loader = DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=1, shuffle=False
    )

    function_scaling = np.vectorize(scaling)
    teachermodel.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            # 獲取輸出矩陣
            output = teachermodel(data)
            output_matrix = output.cpu().numpy()
            
            output_matrix = function_scaling(output_matrix)
            output_matrix.astype(np.int32)
            # 存儲模型的最終輸出
            # pickle_store(output_matrix, '{extraction_teacher_filename}/model_output.pkl')
            # np.savetxt(f'{extraction_teacher_filename}/model_output.csv', output_matrix, delimiter=',')
            # 除存成json
            with open(f'{extraction_teacher_filename}/model_output.json', 'w') as f:
                json.dump(output_matrix.tolist(), f)
            
            # 打印輸出矩陣
            print("Output Matrix:\n", output_matrix)
            break  # 只測試一張圖片