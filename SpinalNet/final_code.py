import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
import scipy
import matplotlib.pyplot as plt
from scipy import stats 
import csv
import imageio
import math
import torch.nn.functional as F
import pickle

transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor()])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transforms.ToTensor())


train_set_data = []
index = 0
for image in trainset:
    random = (image[0], image[1]) + (index,) 
    train_set_data.append(random)
    index = index + 1
    
test_set_data = []

num = 0
for image in testset:
    random = (image[0], image[1]) + (num,) 
    test_set_data.append(random)
    num = num + 1


percent = 2500
batch_size = 100
k =10 #total number of class
# Matrix of gradient. row = classes and columns = image
# #Matrix of gradient. row = classes and columns = image
#Matrix of gradient. row = classes and columns = image

def greedy_sampling(unlabel_index, data, train_index_of_data, query):
    new_sample = []
    existing_sample = []
    remaining_sample = []
    
    for idx in train_index_of_data:
        existing_sample.append(train_set_data[idx])    
    
    if query == "grad_input":
        grad = data[0]
        c_i = data[1]
        class_order = data[2]
        npArray_grad = np.array(grad)
        npArray_c_i = np.array(c_i)
        npArray_class_order = np.array(class_order)
        
        mul = npArray_grad*npArray_c_i
        score = np.sum(mul, axis = 0)
        sorted_index = sorted(range(len(score)), key=lambda k: score[k])
        for sort_idx in sorted_index[:2500]:
            new_sample.append(train_set_data[unlabel_index[sort_idx]])
   
    if query == "max-ent":
        sorted_index = sorted(range(len(data)), key=lambda k: data[k])
        for sort_idx in sorted_index[:2500]:
            new_sample.append(train_set_data[unlabel_index[sort_idx]])

    if query == "margin":
        sorted_index = sorted(range(len(data)), key=lambda k: data[k])
        for sort_idx in sorted_index[:2500]:
            new_sample.append(train_set_data[unlabel_index[sort_idx]])  
    
    remaining_idx = sorted_index[2500:]

    for idx in remaining_idx:
        remaining_sample.append(train_set_data[idx])

    new_sample.extend(existing_sample)
    return new_sample, remaining_sample 
    


class Data(torch.utils.data.Dataset):
    def __init__(self, dataset, left_dataset,train_or_test,transform=None):
                super(Data, self).__init__()
                if train_or_test == 'train':
                        self.dataset = dataset
                        self.length = len(dataset)
                if train_or_test == 'test':
                        self.dataset = test_set_data
                        self.length = len(test_set_data)
                if train_or_test == 'train_unlabel':
                        self.dataset = left_dataset 
                        self.length = len(left_dataset)
    def __len__(self):
                return self.length
    
    def __getitem__(self, idx):
                image = self.dataset[idx][0] 
                label = self.dataset[idx][1] 
                index = self.dataset[idx][2] 
                sample = {'image' : image, 
                          'label' : label,
                          'index' : index}
        
                #if self.transform:
                #     sample = self.transform(sample)
                return sample



# function will return indices list for a batch 
def batch_label(output, k):
    sorting, indices = torch.sort(output, descending=True) # sorting stores the sorting probabolities and indices stores the class in descending order 
    indices = indices.cpu().numpy() # converting tensor array to numpy array  
    indices = indices[:,:k] # slicing the indices
    indices = np.transpose(indices) # indices dimension = column = image and row = indices in descending order
    sorting = sorting.detach().cpu().numpy()
    sorting = np.transpose(sorting)
    return indices , sorting

# Stores the gradient, probablity and class order in matrix. 
# grad = gradient of complete batch, output = model output, image_index = list of index of images, count = keeps the record of class 
def grad_mat(grad, batch_size):
    grad_list = []
    for batch_number in range(batch_size): 
        mul = torch.norm(grad[batch_number], 2) # calculates the norm of gradient of image with perticular class
        grad_list.append(mul)
    return grad_list


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'

# Hyper-parameters
num_epochs = 1
learning_rate = 0.0001

Half_width =256
layer_width=512

torch.manual_seed(0)



def conv3x3(in_channels, out_channels, stride=1):
    """3x3 kernel size with padding convolutional layer in ResNet BasicBlock."""
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

    
class SpinalVGG(nn.Module):

    def __init__(self, features, num_class=10):
        super().__init__()
        self.features = features
        
        self.fc_spinal_layer1 = nn.Sequential(
            nn.Dropout(), nn.Linear(Half_width, layer_width),
            nn.ReLU(inplace=True),
            )
        self.fc_spinal_layer2 = nn.Sequential(
            nn.Dropout(), nn.Linear(Half_width + layer_width, layer_width),
            nn.ReLU(inplace=True),
            )
        self.fc_spinal_layer3 = nn.Sequential(
            nn.Dropout(), nn.Linear(Half_width + layer_width, layer_width),
            nn.ReLU(inplace=True),
            )
        self.fc_spinal_layer4 = nn.Sequential(
            nn.Dropout(), nn.Linear(Half_width + layer_width, layer_width),
            nn.ReLU(inplace=True),
            )
        self.fc_out = nn.Sequential(
            nn.Dropout(), nn.Linear(layer_width*4, num_class)            
            )


    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        x = output
        x1 = self.fc_spinal_layer1(x[:, 0:Half_width])
        x2 = self.fc_spinal_layer2(torch.cat([ x[:,Half_width:2*Half_width], x1], dim=1))
        x3 = self.fc_spinal_layer3(torch.cat([ x[:,0:Half_width], x2], dim=1))
        x4 = self.fc_spinal_layer4(torch.cat([ x[:,Half_width:2*Half_width], x3], dim=1))
        
        x = torch.cat([x1, x2], dim=1)
        x = torch.cat([x, x3], dim=1)
        x = torch.cat([x, x4], dim=1)
        
        x = self.fc_out(x)
    
        return x
    

def make_layers(cfg, batch_norm=False):
    layers = []

    input_channel = 3
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]

        if batch_norm:
            layers += [nn.BatchNorm2d(l)]
        
        layers += [nn.ReLU(inplace=True)]
        input_channel = l
    
    return nn.Sequential(*layers)



def Spinalvgg11_bn():
    return SpinalVGG(make_layers(cfg['A'], batch_norm=True))

def Spinalvgg13_bn():
    return SpinalVGG(make_layers(cfg['B'], batch_norm=True))

def Spinalvgg16_bn():
    return SpinalVGG(make_layers(cfg['D'], batch_norm=True))

def Spinalvgg19_bn():
    return SpinalVGG(make_layers(cfg['E'], batch_norm=True))



# For updating learning rate
def update_lr(optimizer, lr):    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Train the model

def train_test(train_loader, test_loader, test_unlabel_loader, queries): 

    total_step = len(train_loader)
    curr_lr = learning_rate
    model = Spinalvgg11_bn().to(device)



    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train the model
    total_step = len(train_loader)

    best_accuracy = 0.0

    train_index = []
    for epoch in range(num_epochs):
        for i, data in enumerate(train_loader):
            images, labels, image_idx = data['image'], data['label'], data['index']
            images = images.to(device)
            labels = labels.to(device)
            train_index.extend(image_idx)    
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()        

            if i == 249:
                print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
                


            
        # Test the model
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for i,data in enumerate(test_loader):
                images, labels, idx_test = data['image'], data['label'], data['index']
                images = images.to(device)
                labels = labels.to(device)                
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            if best_accuracy> correct / total:
                curr_lr = learning_rate*np.asscalar(pow(np.random.rand(1),3))
                update_lr(optimizer, curr_lr)
                print('Test Accuracy of NN: {} % Best: {} %'.format(100 * correct1 / total, 100*best_accuracy))
            else:
                best_accuracy = correct / total
                net_opt = model
                print('Test Accuracy of NN: {} % (improvement)'.format(100 * correct / total))
                        
            model.train()
            
    unlabel_index = []
    if queries == "grad_input":    
        grad = [[] for i in range(10)]
        c_i = [[] for i in range(10)]
        class_order = [[] for i in range(10)]
        
    if queries == "max_ent":
        max_ent_list = []

    if queries == "margin":
        margin_list = []    

    for i, data in enumerate(test_unlabel_loader):
        inputs, labels, image_index = data['image'], data['label'], data['index']
        inputs = inputs.to(device)
        labels = labels.to(device)
        unlabel_index.extend(image_index)
        inputs.requires_grad = True
        optimizer.zero_grad()
        image_index = image_index.to(device)
        output = model(inputs)
        softmax = nn.Softmax(dim=1)
        output = softmax(output) 
        indices , sorting = batch_label(output,k)
        if queries == "grad_input":
            count = 0
            for index in indices:
                model.zero_grad()
                index = torch.from_numpy(index).to(device)
                loss = criterion(output, index)
                grad_input = torch.autograd.grad(loss, inputs,retain_graph = True, allow_unused=True)# Calculating gradient of complete batch
                grad_batch = grad_mat(grad_input[0], batch_size)
                grad[count].extend(grad_batch)
                c_i[count].extend(sorting[count])
                class_order[count].extend(index) 
                count = count + 1

        if queries == "max_ent":
            max_ent_batch = maximum_entropy(output)
            max_ent_list.append(max_ent_batch)


        if queries == "margin":
            margin_batch = margin(output)
            margin_list.append(margin_batch)


    if queries == "grad_input":
        complete = [grad, c_i, class_order]
        return complete ,train_index, unlabel_index

    if queries == "max_ent":
        return max_ent_list, train_index, unlabel_index

    if queries == "margin":
        return margin_list, train_index, unlabel_index        

  
initial_training_sample = train_set_data[:2500]
initial_untrain_sample = train_set_data[2500:]

train_data = Data(initial_training_sample,initial_untrain_sample,train_or_test = "train")

test_data = Data(initial_training_sample,initial_untrain_sample,train_or_test = "test")

train_unlabel_set = Data(initial_training_sample,initial_untrain_sample,train_or_test = "train_unlabel")



# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                           batch_size=100, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                          batch_size=100, 
                                          shuffle=False)

train_unlabel_loader = torch.utils.data.DataLoader(dataset=train_unlabel_set,
                                          batch_size=100, 
                                          shuffle=False)


data, train_index, unlabel_index = train_test(train_loader, test_loader, train_unlabel_loader, queries = "grad_input") 


for i in range(19):
    
    new_sample, remaining_set = greedy_sampling(unlabel_index, data, train_index, query = "grad_input")
    
    train_data = Data(new_sample,remaining_set,train_or_test = "train")

    test_data = Data(new_sample,remaining_set,train_or_test = "test")

    train_unlabel_set = Data(new_sample,remaining_set,train_or_test = "train_unlabel")

    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                           batch_size=100, 
                                           shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                          batch_size=100, 
                                          shuffle=False)

    train_unlabel_loader = torch.utils.data.DataLoader(dataset=train_unlabel_set,
                                          batch_size=100, 
                                          shuffle=False)

    print('start training')
    data, train_index, unlabel_index = train_test(train_loader, test_loader, train_unlabel_loader,queries = "grad_input" ) 

    print(i)


