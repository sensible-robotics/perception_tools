import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import datasets, transforms

import matplotlib.pyplot as plt

__all__ = ["get_accuracy", "plot_losses", "train", "validate", "training_loop"]

#############################################
def get_accuracy(_model, _data_loader, device):
   '''
   Function for computing the accuracy of the predictions over the entire 
   data_loader
   '''

   correct_pred = 0
   n = 0

   with torch.no_grad():
      _model.eval()
      for X, y_true in _data_loader:
         X = X.to(device)
         y_true = y_true.to(device)

         _, y_prob = _model(X)
         _, predicted_labels = torch.max(y_prob, 1)

         n+= y_true.size(0)
         correct_pred += (predicted_labels == y_true).sum()

   return correct_pred.float() / n

############################################
def plot_losses(_train_losses, _valid_losses):
   '''
   Function for plotting training and validation losses
   '''

   plt.style.use('seaborn')
   train_losses = np.array(_train_losses)
   valid_losses = np.array(_valid_losses)

   fig, ax = plt.subplots(figsize = (8, 4.5))

   ax.plot(_train_losses, color='blue', label= 'Training loss')
   ax.plot(_valid_losses, color='red', label='Validation loss')
   ax.set(title='Loss over epochs',
          xlabel = 'Epoch',
          ylabel = 'Loss')
   ax.legend()
   fig.show()

   # Change the plot style to default
   plt.style.use('default')

############################################
# Train
def train(_train_loader, _model, _criterion, _optimizer, _device):
   '''
   Function for the training step of the training loop
   '''
   _model.train()
   running_loss = 0
   
   for X, y_true in _train_loader:
     _optimizer.zero_grad()
     
     X = X.to(_device)
     y_true = y_true.to(_device)
     
     # Forward pass
     y_hat, _ = _model(X)
     loss = _criterion(y_hat, y_true)
     running_loss += loss.item() * X.size(0)
     
     # Backward pass
     loss.backward()
     _optimizer.step()
     
   epoch_loss = running_loss / len(_train_loader.dataset)
   return _model, _optimizer, epoch_loss


##########################
# validate
def validate(valid_loader, model, criterion, device):
   ''' 
   Function for the validation step of the training loop
   '''
   model.eval()
   running_loss = 0

   for X, y_true in valid_loader:
      X = X.to(device)
      y_true = y_true.to(device)

      # Forward pass and record loss
      y_hat, _ = model(X)
      loss = criterion(y_hat, y_true)
      running_loss += loss.item() * X.size(0)

   epoch_loss = running_loss / len(valid_loader.dataset)

   return model, epoch_loss

# Training loop
def training_loop(model, criterion, optimizer, train_loader,
                  valid_loader, epochs, device, print_every=1):
   '''
   Function defining the entire training loop
   '''

   # Set objects for storing metrics
   best_loss = 1e10
   train_losses = []
   valid_losses = []

   # Train model
   for epoch in range(0, epochs):
      # Training
      model, optimizer, train_loss = train(train_loader, model, criterion, optimizer, device)

      # Validation
      with torch.no_grad():
         model, valid_loss = validate(valid_loader, model, criterion, device)
         valid_losses.append(valid_loss)

      if epoch % print_every == (print_every - 1):
         train_acc = get_accuracy(model, train_loader, device=device)
         valid_acc = get_accuracy(model, valid_loader, device=device)

         print(f'{datetime.now().time().replace(microsecond=0)} --- '
               f'Epoch: {epoch}\t'
               f'Train loss: {train_loss: .4f}\t'
               f'Valid loss: {valid_loss: .4f}\t'
               f'Train accuracy: {100*train_acc: .2f}\t'
               f'Valid accuracy: {100*valid_acc: .2f}')

   plot_losses(train_losses, valid_losses)

   return model, optimizer, (train_losses, valid_losses)


