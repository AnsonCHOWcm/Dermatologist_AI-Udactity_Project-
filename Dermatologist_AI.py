#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 22:39:50 2021

@author: ccm
"""

import os
from torchvision import datasets
import numpy as np
import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import pandas as pd

#Loading The Dataset 

transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
        mean = [0.485 , 0.456 , 0.406],
        std = [0.229 , 0.224 , 0.225])
    ])

transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
        mean = [0.485 , 0.456 , 0.406],
        std = [0.229 , 0.224 , 0.225])
    ])

training_set = datasets.ImageFolder("./data/train", transform=transform_train)
testing_set = datasets.ImageFolder("./data/test", transform=transform)
validing_set = datasets.ImageFolder("./data/valid", transform=transform)

trainloader = torch.utils.data.DataLoader(training_set, batch_size=64 , shuffle = True)
testloader = torch.utils.data.DataLoader(testing_set , batch_size=64, shuffle = True)
validloader = torch.utils.data.DataLoader(validing_set, batch_size=64, shuffle = True)

loaders = {'train' : trainloader ,
                   'test' : testloader ,
                   'valid' : validloader}


## Define a function for training the model

def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path):
    """returns trained model"""
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf 
    
    for epoch in range(1, n_epochs+1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        
        ###################
        # train the model #
        ###################
        model.train()
        for batch_idx, (data, target) in enumerate(loaders['train']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## find the loss and update the model parameters accordingly
            ## record the average training loss, using something like
            ## train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            
            optimizer.zero_grad()
            
            output = model(data)
            loss = criterion(output , target)
            loss.backward()
            optimizer.step()
            
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            
        ######################    
        # validate the model #
        ######################
        model.eval()
        for batch_idx, (data, target) in enumerate(loaders['valid']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## update the average validation loss
            output = model(data)
            loss = criterion(output , target)
            
            valid_loss  = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))
            

            
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, 
            train_loss,
            valid_loss
            ))
        
        ## TODO: save the model if validation loss has decreased
        if valid_loss < valid_loss_min :
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
            torch.save(model.state_dict(), save_path)
            valid_loss_min = valid_loss
            
    # return trained model
    return model

## Define a function for testing

def test(loaders, model, criterion, use_cuda):

    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.

    model.eval()
    for batch_idx, (data, target) in enumerate(loaders['test']):
        # move to GPU
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update average test loss 
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
        # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1]
        # compare predictions to true label
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)
            
    print('Test Loss: {:.6f}\n'.format(test_loss))

    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))


use_cuda = torch.cuda.is_available()

##Specify model architecture 

model_transfer = models.resnet50(pretrained=True)

for param in model_transfer.parameters():
    param.requires_grad = False
    
model_transfer.fc = nn.Sequential(nn.Linear(2048, 512),
                                  nn.ReLU(),
                                  nn.Dropout(0.5),
                                  nn.Linear(512 , 3),
                                  nn.LogSoftmax(dim = 1)
                                          )



if use_cuda:
    model_transfer = model_transfer.cuda()
    

criterion_transfer = nn.NLLLoss()
optimizer_transfer = optim.RMSprop(model_transfer.fc.parameters(), lr = 0.001)

scheduler = optim.lr_scheduler.StepLR(optimizer_transfer, step_size=100, gamma=0.9)

# train the model
model_transfer =  train(10, loaders, model_transfer, optimizer_transfer, criterion_transfer, use_cuda, 'model_transfer.pt')

# load the model that got the best validation accuracy (uncomment the line below)
#model_transfer.load_state_dict(torch.load('model_transfer.pt'))

model_transfer.load_state_dict(torch.load('model_transfer.pt'))

## Test the model

test(loaders ,model_transfer, criterion_transfer, use_cuda )

## Predict the probability

pred = []

for img in testing_set:
    
    img = torch.unsqueeze(img[0],0)
    
    ps = torch.exp(model_transfer(img))
    
    pred.append((ps.detach().numpy())[0])
    
pred_df = pd.DataFrame(pred)

sample_df = pd.read_csv('sample_predictions.csv')

performance_df = pred_df.drop(1 , axis = 1)

performance_df.columns = ['task_1' , 'task_2']

performance_df.to_csv("performance.csv")
    
    

