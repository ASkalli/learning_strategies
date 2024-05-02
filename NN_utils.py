"""
This script defines the NN class we want to optimize with online training

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#small convnet to save on parameters
class Tiny_convnet(nn.Module):
      
    def __init__(self):
        super(Tiny_convnet, self).__init__()
            
        # Convolutional layer (sees 28x28x1 image tensor)
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5, stride=1, padding=2)  # Output: 28x28x8
        # Max Pooling layer (reduces size to 14x14x8)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 14x14x8
        # Convolutional layer (sees 14x14x8 image tensor)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=2)  # Output: 14x14x16
        # Max Pooling layer (reduces size to 7x7x16)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 7x7x16
        # Fully connected layer, adjusted for the output size of the last max pooling layer
        self.fc = nn.Linear(7 * 7 * 16, 10)  # 10 output classes
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)  # Apply first max pooling
        x = F.relu(self.conv2(x))
        x = self.pool2(x)  # Apply second max pooling
        # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x
    
    def count_parameters(self):
        """
        Counts the number of trainable parameters in a PyTorch model.

        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_params(self):
        """
        method that returns parameters of the model as a 1D array.
        """
        params_list = []
        for param in self.parameters():
            #view(-1) flattens the tensor
            params_list.append(param.view(-1))
            
        full_params = torch.cat(params_list)
        return full_params
    
    def set_params(self, params_to_send):
        """
        Method to set parameters in the neural network for online training.
        params_to_send is a flattened array of parameters.
        """
        current_idx = 0
        for param in self.parameters():
            param_numel = param.data.numel()
            #we get the shape of the parameter and then copy the data from the flattened array
            new_param =  torch.reshape(torch.from_numpy(params_to_send[current_idx: current_idx + param_numel ]),shape=param.data.shape)
            param.data.copy_(new_param)
            current_idx += param_numel
 
    def forward_pass_params(self, params_to_send, X):
        """
        This method is a forward pass that also takes in the parameters of the neural network as a variable,
        to use in online learning.
        """
        self.set_params(params_to_send)
        logits = self.forward(X)
        return logits
    

#this is a simple single layer feed forward NN with ReLU activation and adjustable hidden layer size
class Neural_Net(nn.Module):
    def __init__(self, input_size, hidden_size, n_classes):
        "init method that defines the NN architecture and inherits from nn.Module"
        super(Neural_Net, self).__init__()
        
        self.NN_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_classes),
            #nn.Softmax(dim=1)
        )
        
        self.num_params = sum(p.data.numel() for p in self.parameters())
    
    def forward(self, X):
        "forward pass"
        logits = self.NN_stack(X)
        return logits
    
    
    def reset_weights(self):
        "method to reset the weights of the NN"
        for layer in self.NN_stack:
            if isinstance(layer,nn.Linear):
                layer.reset_parameters()
        for param in self.parameters():
            param.requires_grad = True
                
    def get_params(self):
        "Method to get parameters from the neural network"
        params_list = []
        
        for param in self.parameters():
            params_list.append(param.view(-1))
        
        full_params = torch.cat(params_list)
        return full_params
    
    def set_params(self,params_to_send):
        "Method to set parameters params in the neural network for online training. params_to_send is a column vector "
        idx_prev = 0
        for param in self.parameters():
           n_params = param.data.numel()
           new_param =  torch.reshape(torch.from_numpy(params_to_send[idx_prev: idx_prev + n_params ]),shape=param.data.shape)
           param.data.copy_(new_param)
           idx_prev += n_params
    
    def forward_pass_params(self,params_to_send,X):
        "This method is a forward pass that also takes in the parameters of the neural network as a variable, to use in online learning"
        self.set_params(params_to_send)
        logits = self.NN_stack(X)
        return logits
        

#class for a linear classifier

class Lin_classifier(nn.Module):
    def __init__(self, input_size, n_classes):
        super(Lin_classifier,self).__init__()
        
        self.NN_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, n_classes),
        )
        
    def forward(self,X):
        logits = self.NN_stack(X)
        return logits
    

#class  for building train and test loaders so we can use other datasets with pytorch loop
class Custom_dataset(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def train_pytorch_NN(model, n_epochs, train_loader, test_loader, loss, optimizer):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    print(f"Using {device} device")
    print(model)
    
    #array to store the accuracy of the model
    accuracy_list = []
    for epoch in range(n_epochs):
        model.train()
        for i, (images,labels) in enumerate(train_loader):
            #move data to gpu for faster processing
            images = images.to(device)
            labels = labels.to(device)
            #forward pass
            Y_pred = model.forward(images)
            loss_value = loss(Y_pred,labels)
            #backward pass
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
            
            
            #print accuracy every 100 steps for the test set
            if (i+1) % 100 == 0:
                model.eval()
                correct = 0 
                total = 0
                for images, labels in test_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    Y_pred = model.forward(images)
                    _, predicted = torch.max(Y_pred.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                accuracy = ( 100*correct/total)
                accuracy_list.append(accuracy)
                print(f'Epoch [{epoch+1}/{n_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss_value.item()}, Test Accuracy: {accuracy}%')
    return accuracy_list


def train_online_pop_NN(model, n_epochs, train_loader, test_loader, loss, optimizer):
    "function to train a model using the population based training algorithm,  returns the accuracy and best reward lists"
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    print(f"Using {device} device")
    print(model)
    
    best_reward = []
    accuracy_list = []
    for epoch in range(n_epochs):
        model.eval()
        for i, (features,labels) in enumerate(train_loader):
            
            coordinates = optimizer.ask()
            rewards_list = []
            for k in range(coordinates.shape[0]):
                if device == 'cuda':
                    features = features.to(device)
                    labels = labels.to(device)
                    Y_pred = model.forward_pass_params(coordinates[k,:],features)
                if device == 'cpu':
                    Y_pred = model.forward_pass_params(coordinates[k,:],features)    
                loss_value = loss(Y_pred,labels)
                rewards_list.append(loss_value.detach().cpu().item())
            
            rewards = np.array(rewards_list)[:,np.newaxis]
            optimizer.tell(rewards)
            best_params = coordinates[np.argmin(rewards),:]
            best_reward.append(np.min(rewards))
            print('\r{i+1}',end='')
            #print accuracy every 100 steps for the test set
            if (i+1) % 50 == 0:
                model.eval()
                correct = 0 
                total = 0
                for features, labels in test_loader:
                    features = features.to(device)
                    labels = labels.to(device)
                    Y_pred = model.forward_pass_params(best_params,features)
                    loss_value = loss(Y_pred,labels)
                    _, predicted = torch.max(Y_pred.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                accuracy = ( 100*correct/total)
                accuracy_list.append(accuracy)
                print(f'Epoch [{epoch+1}/{n_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss_value.item()}, Test Accuracy: {accuracy}%')
    return accuracy_list, best_reward
    

def train_online_SPSA_NN(model, n_epochs, train_loader, test_loader, loss, spsa_optimizer,adam_optimizer):
    "function to train a model using the population based training algorithm,  returns the accuracy and best reward lists"
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    print(f"Using {device} device")
    print(model)
    
    best_reward = []
    accuracy_list = []
    for epoch in range(n_epochs):
        model.eval()
        for i, (features,labels) in enumerate(train_loader):
            
            params_plus,params_minus = spsa_optimizer.perturb_parameters()
            
            
            if device == 'cuda':
                features = features.to(device)
                labels = labels.to(device)
                Y_pred_plus = model.forward_pass_params(params_plus,features)
                Y_pred_minus = model.forward_pass_params(params_minus,features)
            if device == 'cpu':
                Y_pred_plus = model.forward_pass_params(params_plus,features)
                Y_pred_minus = model.forward_pass_params(params_minus,features)
                
            loss_value_plus = loss(Y_pred_plus,labels)
            loss_value_minus = loss(Y_pred_minus,labels)
            
            reward_plus = loss_value_plus.detach().cpu().item()
            reward_minus = loss_value_minus.detach().cpu().item()
            
            grad_spsa = spsa_optimizer.approximate_gradient(reward_plus ,reward_minus)
            step = adam_optimizer.step(grad_spsa)
            
            current_params= spsa_optimizer.update_parameters_step(step)

            best_reward.append(np.min([reward_plus,reward_minus]))
            print('\r{i+1}',end='')
            #print accuracy every 100 steps for the test set
            if (i+1) % 100 == 0:
                model.eval()
                correct = 0 
                total = 0
                for features, labels in test_loader:
                    features = features.to(device)
                    labels = labels.to(device)
                    Y_pred = model.forward_pass_params(current_params,features)
                    loss_value = loss(Y_pred,labels)
                    _, predicted = torch.max(Y_pred.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                accuracy = ( 100*correct/total)
                accuracy_list.append(accuracy)
                print(f'Epoch [{epoch+1}/{n_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss_value.item()}, Test Accuracy: {accuracy}%')
    return accuracy_list, best_reward
    



def quantize_model(model,quant_levels):
    "function to quantize weights of model on a layer by layer basis, quant_levels is the number of quantization steps"
    with torch.no_grad():
            
        for param in model.parameters():
            min_param = param.min()
            max_param = param.max()
            step = (max_param - min_param ) / (quant_levels)
            
            n_steps = ((param - min_param) / step).round()
            
            quantized_value = min_param + step * n_steps
            
            param.copy_(quantized_value)
    return
