# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 16:51:47 2023

@author: Leonid
"""

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.tensorboard import SummaryWriter
import math
import matplotlib.pyplot as plt
from datetime import datetime

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.fc1 = nn.Linear(28*28, 128)
		self.fc2 = nn.Linear(128, 128)
		self.fc3 = nn.Linear(128, 10)

	def forward(self, x):
		x = x.view(-1, 28*28)
		x = torch.relu(self.fc1(x))
		x = torch.relu(self.fc2(x))
		x = self.fc3(x)
		return x
    
# Define custom optimizer
class MyAdam(torch.optim.Adam):
	def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), weight_decay=0):
		super().__init__(params, lr=lr, betas=betas)
		self.weight_decay = weight_decay

	def step(self):
		for group in self.param_groups:
			for p in group['params']:
				if p.grad is None:
					continue
				grad = p.grad.data
				if grad.is_sparse:
					raise RuntimeError("Adam does not support sparse gradients")

				state = self.state[p]

				# State initialization
				if len(state) == 0:
					state["step"] = 0
					# Exponential moving average of gradient values
					state["exp_avg"] = torch.zeros_like(p.data)
					# Exponential moving average of squared gradient values
					state["exp_avg_sq"] = torch.zeros_like(p.data)

				exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
				beta1, beta2 = group["betas"]

				state["step"] += 1

				if self.weight_decay != 0:
					grad = grad.add(p.data, alpha=self.weight_decay)

				# Decay the first and second moment running average coefficient
				exp_avg.mul_(beta1).add_(1 - beta1, grad)
				exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

				denom = exp_avg_sq.sqrt().add_(group["eps"])

				bias_correction1 = 1 - beta1 ** state["step"]
				bias_correction2 = 1 - beta2 ** state["step"]
				step_size = group["lr"] * math.sqrt(bias_correction2) / bias_correction1

				p.data.addcdiv_(-step_size, exp_avg, denom)


		

if __name__=='__main__':

    #загружаем данные по цифрам  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = MNIST(root='.', train=True, download=True, transform=ToTensor())
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    # dataloader.dataset
    
    #визуализируем данные
    sample_idx = torch.randint(len(dataloader), size=(1,)).item()
    len(dataloader)
    for i, batch in enumerate(dataloader):
    	figure = plt.figure(figsize=(16, 16))
    	img, label = batch
    	for j in range(img.shape[0]):
    		figure.add_subplot(8, 8, j+1)
    		plt.imshow(img[j].squeeze(), cmap="gray")
    		plt.title(label[j])
    		plt.axis("off")
    		
    	plt.show()
    	break
    
    # Model
    model = Net().to(device)
    
    # Loss functions
    loss_fn = nn.CrossEntropyLoss()
    # Optimizer
    optimizer = MyAdam(model.parameters(), weight_decay=0.00001)
    t0 = datetime.now()
    # Training loop
    num_epochs = 10
    for i in range(num_epochs):
    	for inputs, labels in dataloader:
    		inputs, labels = inputs.to(device), labels.to(device)
    		outputs = model(inputs)
    		loss = loss_fn(outputs, labels)
    
    		optimizer.zero_grad()
    		loss.backward()
    		optimizer.step()
    		#scheduler.step()
    		
    	plt.plot(i,loss.item(),'ro-')
    	print(i,'>> Loss :', loss.item())
    t1 = datetime.now()
    print('время обучения: ', t1-t0)
    plt.title('Losses over iterations')
    plt.xlabel('iterations')
    plt.ylabel('Losses')
    plt.show()
    

