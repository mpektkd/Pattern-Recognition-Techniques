import os
import copy
import re
import sys
from glob import glob
import warnings
import time

import pickle
from google.colab import load_ipython_extension
# import optuna

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import seaborn as sns

import librosa.display

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.metrics import classification_report, confusion_matrix

from scipy import stats

import torch
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader
import torch.optim as optim
from torch import nn
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torch.nn import init
import torch.nn.functional as F
from drive.MyDrive.fast_soft_sort.pytorch_ops import soft_rank

warnings.filterwarnings('ignore') # ignore warning messeges

class SpearmanLoss(nn.Module):

    def __init__(self, regularization="l2", regularization_strength=1.0):
        super(SpearmanLoss, self).__init__()  
        self.regularization = regularization
        self.regularization_strength = regularization_strength
        self.criterion = nn.MSELoss()

    def forward(self, pred, target):
      # fast_soft_sort uses 1-based indexing, divide by len to compute percentage of rank
        # print(pred.cpu().reshape(1, -1).shape)
        self.mse = self.criterion(pred, target)
        pred = soft_rank(
            pred.cpu().reshape(1, -1),
            regularization = self.regularization,
            regularization_strength = self.regularization_strength,
        )
        # print(pred.shape)
        pred = pred.cuda()
        return self.corrcoef(pred / pred.shape[-1], target)
        
    def corrcoef(self, pred, target):
        # np.corrcoef in torch from @mdo
        # https://forum.numer.ai/t/custom-loss-functions-for-xgboost-using-pytorch/960
        pred_n = pred - pred.mean()
        target_n = target - target.mean()
        pred_n = pred_n / pred_n.norm()
        target_n = target_n / target_n.norm()

        return -0.1*(pred_n * target_n).sum() + self.mse

class MultiTaskLoss(nn.Module):
    def __init__(self, task_num):
        super(MultiTaskLoss, self).__init__()
        self.task_num = task_num
        self.log_vars = nn.Parameter(torch.zeros((task_num)))
        self.log_vars_init = self.log_vars.detach().clone()
        self.loss = nn.MSELoss()

    def forward(self, preds, valence, energy, danceability):

        loss0 = self.loss(preds[:, 0], valence)
        loss1 = self.loss(preds[:, 1], energy)
        loss2 = self.loss(preds[:, 2], danceability)

        precision0 = torch.exp(-self.log_vars[0])
        loss0 = precision0*loss0 + self.log_vars[0]

        precision1 = torch.exp(-self.log_vars[1])
        loss1 = precision1*loss1 + self.log_vars[1]

        precision2 = torch.exp(-self.log_vars[2])
        loss2 = precision2*loss2 + self.log_vars[2]
        
        return loss0+loss1+loss2

    def __str__(self):
      return f"The Loss Convergence: {self.log_vars_init.data} -> {self.log_vars.data}"
      
class MultiTaskLossTrain(nn.Module):
    def __init__(self, task_num, loss):
        super(MultiTaskLossTrain, self).__init__()
        self.task_num = task_num
        self.log_vars = nn.Parameter(torch.zeros((task_num)))
        self.log_vars_init = self.log_vars.detach().clone()
        self.loss = loss

    def forward(self, preds, valence, energy, danceability):

        loss0 = self.loss(preds[:, 0], valence)
        loss1 = self.loss(preds[:, 1], energy)
        loss2 = self.loss(preds[:, 2], danceability)

        precision0 = torch.exp(-self.log_vars[0])
        loss0 = precision0*loss0 + self.log_vars[0]

        precision1 = torch.exp(-self.log_vars[1])
        loss1 = precision1*loss1 + self.log_vars[1]

        precision2 = torch.exp(-self.log_vars[2])
        loss2 = precision2*loss2 + self.log_vars[2]
        
        return loss0+loss1+loss2

    def __str__(self):
      return f"The Loss Convergence: {self.log_vars_init.data} -> {self.log_vars.data}"

class MultiTaskLossConst(nn.Module):
    def __init__(self, task_num, loss, weights):
        super(MultiTaskLossConst, self).__init__()
        self.loss = loss
        self.weights = weights

    def forward(self, preds, valence, energy, danceability):

        loss0 = self.loss(preds[:, 0], valence)
        loss1 = self.loss(preds[:, 1], energy)
        loss2 = self.loss(preds[:, 2], danceability)
        # print(loss0, loss1, loss2)
        return self.weights[0]*loss0 + self.weights[1]*loss1 + self.weights[2]*loss2

class SelfAttention(nn.Module):
    def __init__(self, attention_size, batch_first=False):
        super(SelfAttention, self).__init__()

        self.batch_first = batch_first
        self.attention_weights = Parameter(torch.FloatTensor(attention_size))
        print(self.attention_weights.requires_grad)
        self.softmax = nn.Softmax(dim=-1)

        self.non_linearity = nn.Tanh()

        init.uniform_(self.attention_weights.data, -0.005, 0.005)

    def get_mask(self, attentions, lengths):
        """
        Construct mask for padded itemsteps, based on lengths
        """
        max_len = max(lengths.data)
        mask = Variable(torch.ones(attentions.size())).detach()

        if attentions.data.is_cuda:
            mask = mask.cuda()

        for i, l in enumerate(lengths.data):  # skip the first sentence
            if l < max_len:
                mask[i, l:] = 0
        return mask

    def attention(self, inputs, lengths):

        ##################################################################
        # STEP 1 - perform dot product
        # of the attention vector and each hidden state
        ##################################################################

        # inputs is a 3D Tensor: batch, len, hidden_size
        # scores is a 2D Tensor: batch, len
        scores = self.non_linearity(inputs.matmul(self.attention_weights))
        scores = self.softmax(scores)

        ##################################################################
        # Step 2 - Masking
        ##################################################################

        # construct a mask, based on the sentence lengths
        mask = self.get_mask(scores, lengths)

        # apply the mask - zero out masked timesteps
        masked_scores = scores * mask

        # re-normalize the masked scores
        _sums = masked_scores.sum(-1, keepdim=True)  # sums per row
        scores = masked_scores.div(_sums)  # divide by row sum

        ##################################################################
        # Step 3 - Weighted sum of hidden states, by the attention scores
        ##################################################################

        # multiply each hidden state with the attention weights
        weighted = torch.mul(inputs, scores.unsqueeze(-1).expand_as(inputs))

        # sum the hidden states
        representations = weighted.sum(1).squeeze()

        return representations, scores

class ConvolutionalLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel):
        super(ConvolutionalLayer,self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.float()
        
    def forward(self,x):
        # print(x.shape)
        return self.conv(x)

class FullyConnectedLayer(nn.Module):
    def __init__(self, fc1_out, fc2_in, fc2_out, dropout = None):
        super(FullyConnectedLayer, self).__init__()
        
        self.fc1 = nn.LazyLinear(fc1_out)
        self.fc2 = nn.Linear(fc2_in, fc2_out)
        self.dropout = dropout
        self.float()
          
    def forward(self, x):

        # dense layer-1
        out = self.fc1(x)
        out = F.relu(out)
        out = F. dropout(out, p=self.dropout) if self.dropout is not None else out

        # dense layer-2
        out = self.fc2(out)

        return out
     
class CNN2(nn.Module):

    def __init__(self, layer_channels, hidden_features, out_features, kernels, chroma=False, dropout=0.2):
    
        super(CNN2, self).__init__()
        self.layer_channels = layer_channels[:-3] if chroma else layer_channels

        self.conv_layer = nn.Sequential(
            *[ConvolutionalLayer(in_channels, out_channels, kernel) 
             for (in_channels, out_channels), kernel in zip(self.layer_channels, kernels)]
        )    

        self.dropout=dropout
        self.fc = nn.LazyLinear(out_features)
        
        self.float()
        
    def forward(self, x):
        
        # convolutional layer
        out = self.conv_layer(x)

        # # global average pooling 2D
        # out = F.max_pool2d(out, kernel_size=out.size()[2:])
        # out = out.view(-1, out.size(1))

        # flatten output
        out = out.view(out.size(0), -1)
        
        # Dropout Layer
        out = F.dropout(out, p=self.dropout)

        # fully connected layer 
        out = self.fc(out)

        return out

class CNN(nn.Module):

    def __init__(self, layer_channels, hidden_features, out_features, kernels, chroma=False, dropout=0.2):
    
        super(CNN, self).__init__()
        self.layer_channels = layer_channels[:-3] if chroma else layer_channels

        self.conv_layer = nn.Sequential(
            *[ConvolutionalLayer(in_channels, out_channels, kernel) 
             for (in_channels, out_channels), kernel in zip(self.layer_channels, kernels)]
        )    

        self.fc = FullyConnectedLayer(hidden_features, hidden_features, out_features, dropout=dropout)
        
        self.float()
        
    def forward(self, x):
        
        # convolutional layer
        out = self.conv_layer(x)

        # # global average pooling 2D
        # out = F.max_pool2d(out, kernel_size=out.size()[2:])
        # out = out.view(-1, out.size(1))

        # flatten output
        out = out.view(out.size(0), -1)
        
        # fully connected layer 
        out = self.fc(out)

        return out

class MultiCNN(nn.Module):

    def __init__(self, layer_channels, hidden_features, out_features, kernels, chroma=False, dropout=0.2):
    
        super(MultiCNN, self).__init__()
        self.layer_channels = layer_channels[:-2] if chroma else layer_channels

        self.conv_layer = nn.Sequential(
            *[ConvolutionalLayer(in_channels, out_channels, kernel) 
             for (in_channels, out_channels), kernel in zip(self.layer_channels, kernels)]
        )    


        self.fc_val = FullyConnectedLayer(hidden_features, hidden_features, out_features, dropout=dropout)
        self.fc_en = FullyConnectedLayer(hidden_features, hidden_features, out_features, dropout=dropout)
        self.fc_danc = FullyConnectedLayer(hidden_features, hidden_features, out_features, dropout=dropout)
        
        self.float()
        
    def forward(self, x):
        
        # convolutional layer
        out = self.conv_layer(x)

        # flatten output
        out = out.view(out.size(0), -1)

        # fully connected layer
        valence = self.fc_val(out)
        energy = self.fc_en(out)
        danceability = self.fc_danc(out)

        return torch.cat((valence, energy, danceability), 1)

class LSTM2(SelfAttention):
    def __init__(self, input_dim, rnn_size, output_dim, num_layers, bidirectional=False, dropout=None, pack_padded_sequence=False, att=False):
        super(LSTM2, self).__init__(2 * rnn_size if bidirectional else rnn_size, batch_first=True)
        
        self.att = att
        self.bidirectional = bidirectional
        self.feature_size = rnn_size * 2 if self.bidirectional else rnn_size
        self.num_layers = num_layers
        self.pack_padded_sequence = pack_padded_sequence
        
        #define the lstm
        self.lstm = nn.LSTM(input_dim, self.feature_size, self.num_layers, batch_first=True, dropout=dropout)

        #define a non-linear transformation of the representations
        # self.output = nn.Linear(self.feature_size, output_dim)
        self.fc = FullyConnectedLayer(self.feature_size // 2, self.feature_size // 2, output_dim, dropout=dropout)

        self.float()


    ## TODO: I may have to implement the init_hidden() for initializing h0
    #def init_hidden(self): . . .

    def forward(self, x, lengths):
        """ 
            x : 3D numpy array of dimension N x L x D
                N: batch index
                L: sequence index
                D: feature index
            lengths: N x 1
         """
        #define batch_size and max_length
        batch_size, max_length, _ = x.shape

        # Improving Training Complexity (Bonus)
        if self.pack_padded_sequence:
            
            # Sort by length, pass through LSTM layer as pack_padded_sequence and retieve initial ordering
            # Sort by length (keep idx)
            sorted_lengths, idx_sort = np.sort(lengths)[::-1], np.argsort(-lengths)
            idx_unsort = np.argsort(idx_sort)
  
            # x = x.index_select(0, Variable(idx_sort)) # i may have to use it
            x = x.index_select(0, idx_sort)
  
            # Handling padding in Recurrent Networks
            x = nn.utils.rnn.pack_padded_sequence(x, sorted_lengths.tolist(), batch_first=True) 
            ht, _ = self.lstm(x)
            ht, _ = nn.utils.rnn.pad_packed_sequence(ht, batch_first=True) 
  
            # Un-sort by length
            ht = ht.index_select(0, idx_unsort)
  
          # ht, _ = self.lstm(X, (h_0, c_0), batch_first=True) ## for another verion
        else:
            ht, _ = self.lstm(x)
        # You must have all of the outputs of the LSTM, but you need only the last one (that does not exceed the sequence length)
        # To get it use the last_timestep method
        # Then pass it through the remaining network

        if self.att:
          # apply attention to get spectogram representation
          last_step, _ = self.attention(ht, lengths)
        else:
          # Given implementation
          last_step = self.last_timestep(ht, lengths, self.bidirectional)

        logits = self.fc(last_step) 

        return logits

    def last_timestep(self, outputs, lengths, bidirectional=False):
        """
            Returns the last output of the LSTM taking into account the zero padding
        """
        if bidirectional:
            forward, backward = self.split_directions(outputs)
            last_forward = self.last_by_index(forward, lengths)
            last_backward = backward[:, 0, :]
            
            # Concatenate and return - maybe add more functionalities like average
            return torch.cat((last_forward, last_backward), dim=-1)

        else:
            return self.last_by_index(outputs, lengths)

    @staticmethod
    def split_directions(outputs):
        direction_size = int(outputs.size(-1) / 2)
        forward = outputs[:, :, :direction_size]
        backward = outputs[:, :, direction_size:]
        return forward, backward

    @staticmethod
    def last_by_index(outputs, lengths):
        # Index of the last output for each sequence.
        idx = (lengths - 1).view(-1, 1).expand(outputs.size(0),
                                               outputs.size(2)).unsqueeze(1)
        return outputs.gather(1, idx).squeeze()

class LSTM(nn.Module):
    def __init__(self, input_dim, rnn_size, output_dim, num_layers, bidirectional=False, dropout=None, pack_padded_sequence=False):
        super(LSTM, self).__init__()
        
        self.bidirectional = bidirectional
        self.feature_size = rnn_size * 2 if self.bidirectional else rnn_size
        self.num_layers = num_layers
        self.pack_padded_sequence = pack_padded_sequence
        
        #define the lstm
        self.lstm = nn.LSTM(input_dim, self.feature_size, self.num_layers, batch_first=True)

        #Define dropout layer(because of num_layers==1, we are setting a distinct dropout layer after lstm)
        self.dropout = nn.Dropout(dropout) if dropout is not None else None
        
        #define a non-linear transformation of the representations
        self.output = nn.Linear(self.feature_size, output_dim)

        self.float()


    ## TODO: I may have to implement the init_hidden() for initializing h0
    #def init_hidden(self): . . .

    def forward(self, x, lengths):
        """ 
            x : 3D numpy array of dimension N x L x D
                N: batch index
                L: sequence index
                D: feature index
            lengths: N x 1
         """
        #define batch_size and max_length
        batch_size, max_length, _ = x.shape

        # Improving Training Complexity (Bonus)
        if self.pack_padded_sequence:
            
            # Sort by length, pass through LSTM layer as pack_padded_sequence and retieve initial ordering
            # Sort by length (keep idx)
            sorted_lengths, idx_sort = np.sort(lengths)[::-1], np.argsort(-lengths)
            idx_unsort = np.argsort(idx_sort)
  
            # x = x.index_select(0, Variable(idx_sort)) # i may have to use it
            x = x.index_select(0, idx_sort)
  
            # Handling padding in Recurrent Networks
            x = nn.utils.rnn.pack_padded_sequence(x, sorted_lengths.tolist(), batch_first=True) 
            ht, _ = self.lstm(x)
            ht, _ = nn.utils.rnn.pad_packed_sequence(ht, batch_first=True) 
  
            # Un-sort by length
            ht = ht.index_select(0, idx_unsort)
  
          # ht, _ = self.lstm(X, (h_0, c_0), batch_first=True) ## for another verion
        else:
            ht, _ = self.lstm(x)

        # You must have all of the outputs of the LSTM, but you need only the last one (that does not exceed the sequence length)
        # To get it use the last_timestep method
        # Then pass it through the remaining network

        # Given implementation
        last_step = self.last_timestep(ht, lengths, self.bidirectional)

        logits = self.output(self.dropout(last_step)) if self.dropout is not None else self.output(last_step)

        return logits

    def last_timestep(self, outputs, lengths, bidirectional=False):
        """
            Returns the last output of the LSTM taking into account the zero padding
        """
        if bidirectional:
            forward, backward = self.split_directions(outputs)
            last_forward = self.last_by_index(forward, lengths)
            last_backward = backward[:, 0, :]
            
            # Concatenate and return - maybe add more functionalities like average
            return torch.cat((last_forward, last_backward), dim=-1)

        else:
            return self.last_by_index(outputs, lengths)

    @staticmethod
    def split_directions(outputs):
        direction_size = int(outputs.size(-1) / 2)
        forward = outputs[:, :, :direction_size]
        backward = outputs[:, :, direction_size:]
        return forward, backward

    @staticmethod
    def last_by_index(outputs, lengths):
        # Index of the last output for each sequence.
        idx = (lengths - 1).view(-1, 1).expand(outputs.size(0),
                                               outputs.size(2)).unsqueeze(1)
        return outputs.gather(1, idx).squeeze()

def clf_train(device, _epoch, dataloader, feats, model, loss_function, optimizer, overfit_batch=False, cnn=False):
    
    # IMPORTANT: switch to train mode
    # enable regularization layers, such as Dropout
    model.train()
    
    running_loss = 0.0
    for index, batch in enumerate(dataloader, 1):

        inputs, labels, lengths = batch

        # move the batch tensors to the right device
        inputs, labels, lengths = inputs.to(device), labels.to(device), lengths.to(device)  # EX9
        _, maxseqlen, _ = inputs.shape
        inputs = inputs.reshape(-1, 1, maxseqlen, feats).to(device) if cnn else inputs.to(device)

        # zero the gradients
        # Remember that PyTorch accumulates gradients.
        # We need to clear them out before each batch!
        model.zero_grad() 
        
        # forward pass: y' = model(x)
        outputs = model(inputs.float()) if cnn else model(inputs.float(), lengths)
        
        # compute loss: L = loss_function(y, y')
        loss = loss_function(outputs, labels) 
        
        # backward pass: compute gradient wrt model parameters
        loss.backward()

        # update weights
        optimizer.step()
        
        running_loss += loss.data.item()
        progress(loss=loss.data.item(),
                 epoch=_epoch,
                 batch=index,
                 batch_size=dataloader.batch_size,
                 dataset_size=len(dataloader.dataset))
        if overfit_batch:
          break
        
def clf_eval(device, dataloader, feats, model, loss_function, overfit_batch=False, cnn=False):
    
    # IMPORTANT: switch to eval mode
    # disable regularization layers, such as Dropout
    model.eval()
    
    running_loss = 0.0

    y_pred = []  # the predicted labels
    y = []  # the gold labels

    # IMPORTANT: in evaluation mode, we don't want to keep the gradients
    # so we do everything under torch.no_grad()
    _ = next(model.parameters()).device
    
    with torch.no_grad():
        for index, batch in enumerate(dataloader, 1):
           
            # get the inputs (batch)
            inputs, labels, lengths = batch

            # move the batch tensors to the right device
            inputs, labels, lengths = inputs.to(device), labels.to(device), lengths.to(device)  # EX9
            _, maxseqlen, _ = inputs.shape
            inputs = inputs.reshape(-1, 1, maxseqlen, feats).to(device) if cnn else inputs.to(device)

            # forward pass: y' = model(x)
            outputs = model(inputs.float()) if cnn else model(inputs.float(), lengths) 

            # compute loss
            # We compute the loss only for inspection (compare train/test loss)
            # because we do not actually backpropagate in test time

            loss = loss_function(outputs, labels)
        
            # make predictions (class = argmax of posteriors)
        
            val, pred = outputs.max(1) # argmax since output is a prob distribution  

            # collect the predictions, gold labels and batch loss
            tags = []
      
            y += list(labels)
            y_pred += list(pred) 
            running_loss += loss.data.item()
            
            if overfit_batch:
              break

    return running_loss / index, [x.item() for x in y], [x.item() for x in y_pred]

def reg_train(device, _epoch, dataloader, feats, model, loss_function, optimizer, overfit_batch=False, cnn=False, multi=False):
    
    # IMPORTANT: switch to train mode
    # enable regularization layers, such as Dropout
    model.train()
    
    running_loss = 0.0
    for index, batch in enumerate(dataloader, 1):

        inputs, labels, lengths = batch

        # move the batch tensors to the right device
        inputs, labels, lengths = inputs.to(device), labels.to(device), lengths.to(device) 
        _, maxseqlen, _ = inputs.shape
        inputs = inputs.reshape(-1, 1, maxseqlen, feats).to(device) if cnn else inputs.to(device)

        # zero the gradients
        # Remember that PyTorch accumulates gradients.
        # We need to clear them out before each batch!
        model.zero_grad() 

        # forward pass: y' = model(x)
        outputs = model(inputs.float()) if cnn else model(inputs.float(), lengths)
        # compute loss: L = loss_function(y, y')
        # loss = loss_function(outputs, labels[:, 0], labels[:, 1], labels[:, 2]) if multi else loss_function(outputs, labels)
        loss = loss_function(outputs.double(), labels[:, 0], labels[:, 1], labels[:, 2]) if multi else loss_function(outputs.double(), labels)
        
        # backward pass: compute gradient wrt model parameters
        loss.backward()

        # update weights
        optimizer.step()
        
        running_loss += loss.data.item()
        progress(loss=loss.data.item(),
                 epoch=_epoch,
                 batch=index,
                 batch_size=dataloader.batch_size,
                 dataset_size=len(dataloader.dataset))
        if overfit_batch:
          break

def reg_eval(device, dataloader, feats, model, loss_function, overfit_batch=False, cnn=False, multi=False, titles=None):
    
    # IMPORTANT: switch to eval mode
    # disable regularization layers, such as Dropout
    model.eval()
    
    running_loss = 0.0

    y_pred1 = []  # the predicted labels
    y1 = []  # the gold labels

    y_pred2 = []  # the predicted labels
    y2 = []  # the gold labels
    
    y_pred3 = []  # the predicted labels
    y3 = []  # the gold labels
      
    # IMPORTANT: in evaluation mode, we don't want to keep the gradients
    # so we do everything under torch.no_grad()
    _ = next(model.parameters()).device
    
    with torch.no_grad():
        for index, batch in enumerate(dataloader, 1):
           
            # get the inputs (batch)
            inputs, labels, lengths = batch
            # move the batch tensors to the right device
            inputs, labels, lengths = inputs.to(device), labels.to(device), lengths.to(device)  # EX9
            _, maxseqlen, _ = inputs.shape
            inputs = inputs.reshape(-1, 1, maxseqlen, feats).to(device) if cnn else inputs.to(device)

            # forward pass: y' = model(x)
            outputs = model(inputs.float()) if cnn else model(inputs.float(), lengths) 

            # compute loss
            # We compute the loss only for inspection (compare train/test loss)
            # because we do not actually backpropagate in test time
            loss = loss_function(outputs, labels[:, 0], labels[:, 1], labels[:, 2]) if multi else loss_function(outputs, labels)
        
            # collect the predictions, gold labels and batch loss

            if multi:
              y1 += list(labels[:, 0].cpu())
              y_pred1 += list(outputs[:, 0].cpu()) 

              y2 += list(labels[:, 1].cpu())
              y_pred2 += list(outputs[:, 1].cpu()) 
              
              y3 += list(labels[:, 2].cpu())
              y_pred3 += list(outputs[:, 2].cpu())   

            else:
              y1 += list(labels.cpu())
              y_pred1 += list(outputs.cpu())  

            running_loss += loss.data.item()
            
            if overfit_batch:
              break
    if titles is not None:
      plot_corr([y1, y2, y3], [y_pred1, y_pred2, y_pred3], titles, multi=multi)
    
    return running_loss / index, [y1, y2, y3], [y_pred1, y_pred2, y_pred3]

def reg_main(device, net, feats, optimizer, criterion, train_loader, dev_loader, EPOCHS, Descrs, net_name=None, PATIENCE=None, overfit_batch=False, cnn=False, multi=False):
    # try:
    #############################################################################
    # Training Pipeline
    #############################################################################
    losses = np.zeros((2,EPOCHS))
    
    spear1 = np.zeros((2,EPOCHS))
    spear2 = np.zeros((2,EPOCHS))
    spear3 = np.zeros((2,EPOCHS))

    total = 0
    base = time.time()
    early = EarlyStoppingReg(patience=PATIENCE, multi=multi) if PATIENCE is not None else None
    ep = 0
    E_loss = np.inf
  
    for epoch in tqdm(range(1, EPOCHS + 1)):
        ep += 1
        now = time.time()
        
        # train the model for one epoch
        reg_train(device, epoch, train_loader, feats, net, criterion, optimizer, cnn=cnn, multi=multi)
        # evaluate the performance of the model, on both data sets
        train_loss, y_train_gold, y_train_pred = reg_eval(device, train_loader, feats, net, criterion, cnn=cnn, multi=multi)

        if math.isnan(train_loss):
          losses[0, epoch-1] = losses[0, epoch-2]
          spear1[0, epoch-1] = spear1[0, epoch-2]
          if multi:
            spear2[0, epoch-1] = spear2[0, epoch-2]
            spear3[0, epoch-1] = spear2[0, epoch-2]
          break

        losses[0, epoch-1] = train_loss

        spear1[0, epoch-1] = stats.spearmanr(y_train_pred[0], y_train_gold[0])[0]
        if multi:
          spear2[0, epoch-1] = stats.spearmanr(y_train_pred[1], y_train_gold[1])[0]
          spear3[0, epoch-1] = stats.spearmanr(y_train_pred[2], y_train_gold[2])[0]
  
        print(f"\nStatistics for the Train Set")
        print(f'\t Epoch: {epoch} \t loss: {losses[0, epoch-1]}')
        print(f'\t Epoch: {epoch} \t Spear Corr 1: {spear1[0, epoch-1]}')
        if multi:
          print(f'\t Epoch: {epoch} \t Spear Corr 2: {spear2[0, epoch-1]}')
          print(f'\t Epoch: {epoch} \t Spear Corr 3: {spear3[0, epoch-1]}')

        test_loss, y_test_gold, y_test_pred = reg_eval(device, dev_loader, feats, net, criterion, cnn=cnn, multi=multi)
        
        if math.isnan(test_loss):
          losses[1, epoch-1] = losses[1, epoch-2]
          spear1[1, epoch-1] = spear1[1, epoch-2]
          if multi:
            spear2[1, epoch-1] = spear2[1, epoch-2]
            spear3[1, epoch-1] = spear2[1, epoch-2]
          break
        
        if net_name is not None and E_loss > test_loss:
            E_loss = test_loss
            with open(f'./best_{net_name}.pickle', 'wb') as handle:  # Use pickle files to save our models
                pickle.dump(net, handle, protocol=pickle.HIGHEST_PROTOCOL)
  
        losses[1, epoch-1] = test_loss
        spear1[1, epoch-1] = stats.spearmanr(y_test_pred[0], y_test_gold[0])[0]
        if multi:
          spear2[1, epoch-1] = stats.spearmanr(y_test_pred[1], y_test_gold[1])[0]
          spear3[1, epoch-1] = stats.spearmanr(y_test_pred[2], y_test_gold[2])[0]

        print(f"Statistics for the Dev Set")
        print(f'\t Epoch: {epoch} \t loss: {losses[1, epoch-1]}')
        print(f'\t Epoch: {epoch} \t Spear Corr 1: {spear1[1, epoch-1]}')
        if multi:
          print(f'\t Epoch: {epoch} \t Spear Corr 2: {spear2[1, epoch-1]}')
          print(f'\t Epoch: {epoch} \t Spear Corr 3: {spear3[1, epoch-1]}')

        if early is not None:
            early.__call__(spear1, spear2, spear3, train_loss, test_loss, epoch)   #call the object early for checking the advance
            if early.stopping() == True:    #if true then stop the training to avoid overfitting
                  break
  
        tm = time.time() - now
        total += tm
        print("Epoch total time", tm)
  
    print("Training total time", total)
  
    epochs = np.linspace(1, ep, ep)
    best = early.get_best() if early is not None else None
    # Plot our Study's Result
    plot_statsReg(losses, spear1, spear2, spear3, epochs, ep, Descrs, best=best, multi=multi)
    # except KeyboardInterrupt:
    #   plot_statsReg(losses, spear1, spear2, spear3, epochs, ep, Descrs, best=best, multi=multi)
    #   raise(KeyboardInterrupt('Ctrl C pressed'))
    
def clf_main(device, net, feats, optimizer, criterion, train_loader, dev_loader, EPOCHS, net_name=None, PATIENCE=None, overfit_batch=False, cnn=False):
    # try:
    #############################################################################
    # Training Pipeline
    #############################################################################
    losses = np.zeros((2,EPOCHS))
    accuracy = np.zeros((2,EPOCHS))
    f1 = np.zeros((2,EPOCHS))
    recall = np.zeros((2,EPOCHS))
    total = 0
    base = time.time()
    early = EarlyStopping(patience=PATIENCE) if PATIENCE is not None else None
    ep = 0
    E_loss = np.inf
  
    for epoch in tqdm(range(1, EPOCHS + 1)):
        ep += 1
        now = time.time()
        
        # train the model for one epoch
        clf_train(device, epoch, train_loader, feats, net, criterion, optimizer, cnn=cnn)
        # evaluate the performance of the model, on both data sets
        train_loss, y_train_gold, y_train_pred = clf_eval(device, train_loader, feats, net, criterion, cnn=cnn)
        losses[0, epoch-1] = train_loss
  
        print(f"\nStatistics for the Train Set")
        print(f'\t Epoch: {epoch} \t loss: {losses[0, epoch-1]}')
        accuracy[0, epoch-1] = accuracy_score(y_train_gold, y_train_pred)
        print(f'\t Epoch: {epoch} \t Accuracy Score: {accuracy_score(y_train_gold, y_train_pred)}')
        f1[0, epoch-1] = f1_score(y_train_gold, y_train_pred, average='macro')
        print(f'\t Epoch: {epoch} \t f1 Score: {f1[0, epoch-1]}')
        recall[0, epoch-1] = recall_score(y_train_gold, y_train_pred, average='macro')
        print(f'\t Epoch: {epoch} \t recall Score: {recall[0, epoch-1]}')
        
        test_loss, y_test_gold, y_test_pred = clf_eval(device, dev_loader, feats, net, criterion, cnn=cnn)
        
        if net_name is not None and E_loss > test_loss:
            E_loss = test_loss
            with open(f'./best_{net_name}.pickle', 'wb') as handle:  # Use pickle files to save our models
                pickle.dump(net, handle, protocol=pickle.HIGHEST_PROTOCOL)
  
        losses[1, epoch-1] = test_loss
        print(f"Statistics for the Dev Set")
        print(f'\t Epoch: {epoch} \t loss: {losses[1, epoch-1]}')
  
        accuracy[1, epoch-1] = accuracy_score(y_test_gold, y_test_pred)
        print(f'\t Epoch: {epoch} \t Accuracy Score: {accuracy[1, epoch-1]}')
        f1[1, epoch-1] = f1_score(y_test_gold, y_test_pred, average='macro')
        print(f'\t Epoch: {epoch} \t f1 Score: {f1[1, epoch-1]}')
        recall[1, epoch-1] = recall_score(y_test_gold, y_test_pred, average='macro')
        print(f'\t Epoch: {epoch} \t recall Score: {recall[1, epoch-1]}')
  
        if early is not None:
            early.__call__(accuracy, f1, recall, train_loss, test_loss, epoch)   #call the object early for checking the advance
            if early.stopping() == True:    #if true then stop the training to avoid overfitting
                  break
  
        tm = time.time() - now
        total += tm
        print("Epoch total time", tm)
  
    print("Training total time", total)
  
    epochs = np.linspace(1, ep, ep)
    best = early.get_best() if early is not None else None
    # Plot our Study's Result
    plot_stats(losses, accuracy, f1, recall, epochs, ep, best=best)
    # except KeyboardInterrupt:
    #   plot_stats(losses, accuracy, f1, recall, epochs, ep, best=best)
    #   raise(KeyboardInterrupt('Ctrl C pressed'))

class EarlyStoppingReg:#class for the early stopping reguralization
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=3, verbose=True, delta=0, multi=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 3
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.best = {}
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.dev_loss_min = np.Inf
        self.delta = delta  #definition of the minimum tolerance
        self.multi = multi

    def __call__(self, spear1, spear2, spear3, train_loss, dev_loss, epoch):

        score = -dev_loss

        if self.best_score is None: #check if it is the first epoch

            self.best_score = score
            self.best['epoch'] = epoch
            self.best['loss'] = [train_loss, dev_loss]
            self.best['spear1'] = [spear1[0, epoch-1], spear1[1, epoch-1]]
            if self.multi:
              self.best['spear2'] = [spear2[0, epoch-1], spear2[1, epoch-1]]
              self.best['spear3'] = [spear3[0, epoch-1], spear3[1, epoch-1]]
            self.dev_loss_min = dev_loss

            if self.verbose:
                print(f'Dev loss decreased ({self.dev_loss_min:.6f} --> {dev_loss:.6f}).  Saving model ...')
            

        elif score < self.best_score + self.delta:  #if there is no advance then increase counter
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')

            if self.counter >= self.patience:   #if counter == patience then stop
                self.early_stop = True
        else:
            if self.verbose:
                print(f'Dev loss decreased ({self.dev_loss_min:.6f} --> {dev_loss:.6f}).  Saving model ...')

            self.best_score = score #else save the best model till now
            self.counter = 0
            self.best['epoch'] = epoch
            self.best['loss'] = [train_loss, dev_loss]
            self.best['spear1'] = [spear1[0, epoch-1], spear1[1, epoch-1]]
            if self.multi:
              self.best['spear2'] = [spear2[0, epoch-1], spear2[1, epoch-1]]
              self.best['spear3'] = [spear3[0, epoch-1], spear3[1, epoch-1]]
            self.dev_loss_min = dev_loss

    def stopping(self):
        return self.early_stop

    def get_best(self):
        return self.best

class EarlyStopping():  #class for the early stopping reguralization
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=3, verbose=True, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 3
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.best = {}
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.test_loss_min = np.Inf
        self.delta = delta  #definition of the minimum tolerance

    def __call__(self, accuracy, f1, recall, train_loss, test_loss, epoch):

        score = -test_loss

        if self.best_score is None: #check if it is the first epoch
            self.best_score = score
            self.best['epoch'] = epoch
            self.best['loss'] = [train_loss, test_loss]
            self.best['accuracy'] = [accuracy[0, epoch-1], accuracy[1, epoch-1]]
            self.best['f1'] = [f1[0, epoch-1], f1[1, epoch-1]]
            self.best['recall'] = [recall[0, epoch-1], recall[1, epoch-1]]
            
            if self.verbose:
                print(f'Test loss decreased ({self.test_loss_min:.6f} --> {test_loss:.6f}).  Saving model ...')
            self.test_loss_min = test_loss
        elif score < self.best_score + self.delta:  #if there is no advance then increase counter
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:   #if counter == patience then stop
                self.early_stop = True
        else:
            self.best_score = score #else save the best model till now
            if self.verbose:
                print(f'Test loss decreased ({self.test_loss_min:.6f} --> {test_loss:.6f}).  Saving model ...')
            self.counter = 0
            self.best['epoch'] = epoch
            self.best['loss'] = [train_loss, test_loss]
            self.best['accuracy'] = [accuracy[0, epoch-1], accuracy[1, epoch-1]]
            self.best['f1'] = [f1[0, epoch-1], f1[1, epoch-1]]
            self.best['recall'] = [recall[0, epoch-1], recall[1, epoch-1]]
            self.test_loss_min = test_loss

    def stopping(self):
        return self.early_stop

    def get_best(self):
        return self.best

# TODO: Comment on howv the train and validation splits are created.
# TODO: It's useful to set the seed when debugging but when experimenting ALWAYS set seed=None. Why?
# If the a value is None, then by default, current system time is used.
def torch_train_val_split(
        dataset, batch_train, batch_eval,
        val_size=.2, shuffle=True, seed=None):
    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    val_split = int(np.floor(val_size * dataset_size))
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)

    train_indices = indices[val_split:]
    val_indices = indices[:val_split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset,
                              batch_size=batch_train,
                              sampler=train_sampler)
    val_loader = DataLoader(dataset,
                            batch_size=batch_eval,
                            sampler=val_sampler)
    return train_loader, val_loader


def read_fused_spectrogram(spectrogram_file):
    spectrogram = np.load(spectrogram_file)
    return spectrogram.T


def read_mel_spectrogram(spectrogram_file):
    spectrogram = np.load(spectrogram_file)[:128]
    return spectrogram.T

    
def read_chromagram(spectrogram_file):
    spectrogram = np.load(spectrogram_file)[128:]
    return spectrogram.T

class LabelTransformer(LabelEncoder):
    def inverse(self, y):
        try:
            return super(LabelTransformer, self).inverse_transform(y)
        except:
            return super(LabelTransformer, self).inverse_transform([y])

    def transform(self, y):
        try:
            return super(LabelTransformer, self).transform(y)
        except:
            return super(LabelTransformer, self).transform([y])


# TODO: Comment on why padding is needed
class PaddingTransform(object):
    def __init__(self, max_length, padding_value=0):
        self.max_length = max_length
        self.padding_value = padding_value

    def __call__(self, s):
        if len(s) == self.max_length:
            return s

        if len(s) > self.max_length:
            return s[:self.max_length]

        if len(s) < self.max_length:
            s1 = copy.deepcopy(s)
            pad = np.zeros((self.max_length - s.shape[0], s.shape[1]), dtype=np.float32)
            s1 = np.vstack((s1, pad))
            return s1


class SpectrogramDataset(Dataset):
    def __init__(self, path, class_mapping=None, train=True, max_length=-1, read_spec_fn=read_mel_spectrogram, emotion=None):
        t = 'train' if train else 'test'
        p = os.path.join(path, t)
        self.index = os.path.join(path, "{}_labels.txt".format(t))
        if (t == 'test' and emotion is not None):
            self.files = os.listdir(p)
            labels = [-1]*len(self.files)
        else:
            self.files, labels = self.get_files_labels(self.index, class_mapping) if emotion is None else self.get_files_scores(self.index, emotion)             
        self.feats = [read_spec_fn(os.path.join(p, f)) for f in self.files]
        self.feat_dim = self.feats[0].shape[1]
        self.lengths = [len(i) for i in self.feats]
        self.max_length = max(self.lengths) if max_length <= 0 else max_length
        self.zero_pad_and_stack = PaddingTransform(self.max_length)
        self.label_transformer = LabelTransformer()
        if isinstance(labels, (list, tuple)):
            if (emotion is not None):
              self.labels = np.array(labels).astype('float64')
            else:
              self.labels = np.array(self.label_transformer.fit_transform(labels)).astype('int64')

    def get_files_labels(self, txt, class_mapping):
        with open(txt, 'r') as fd:
            lines = [l.rstrip().split('\t') for l in fd.readlines()[1:]]
        files, labels = [], []
        for l in lines:
            label = l[1]
            if class_mapping:
                label = class_mapping[l[1]]
            if not label:
                continue
            # Kaggle automatically unzips the npy.gz format so this hack is needed
            _id = l[0].split('.')[0]
            npy_file = '{}.fused.full.npy'.format(_id)
            files.append(npy_file)
            labels.append(label)
        return files, labels

    def get_files_scores(self, txt, emotion):
        with open(txt, 'r') as fd:
            lines = [l.rstrip().split(',') for l in fd.readlines()[1:]]
        files, labels = [], []
        for l in lines:
            label = np.array(l)[emotion]
            # Kaggle automatically unzips the npy.gz format so this hack is needed
            _id = l[0]
            npy_file = '{}.fused.full.npy'.format(_id)
            files.append(npy_file)
            labels.append(label)
        return files, labels

    def __getitem__(self, item):
        # TODO: Inspect output and comment on how the output is formatted
        l = min(self.lengths[item], self.max_length)
        return self.zero_pad_and_stack(self.feats[item]), self.labels[item], l 

    def __len__(self):
        return len(self.labels)

def progress(loss, epoch, batch, batch_size, dataset_size):
    """
    Print the progress of the training for each epoch
    """
    batches = math.ceil(float(dataset_size) / batch_size)
    count = batch * batch_size
    bar_len = 40
    filled_len = int(round(bar_len * count / float(dataset_size)))

    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    status = 'Epoch {}, Loss: {:.4f}'.format(epoch, loss)
    _progress_str = "\r \r [{}] ...{}".format(bar, status)
    sys.stdout.write(_progress_str)
    sys.stdout.flush()

    if batch == batches:
        print()

def plot_statsReg(losses, spear1, spear2, spear3, epochs, ep, Descrs, best=None, multi=False):

    figure, axes = plt.subplots(nrows=1, ncols=2, figsize = (10,8))
    titles = ['Loss', 'Spearman Correlation']
    functions = [losses, [spear1, spear2, spear3]] if multi else [losses, [spear1]]
    best_keys = ['loss', ['spear1', 'spear2', 'spear3']] if multi else ['loss', ['spear1']]
    descrs = ['Loss', Descrs]
  
    for ax, title, function, key, descr in zip(axes.flatten(), titles, functions, best_keys, descrs):
        if isinstance(function, list):
          if best is not None:
            lbl=False
            for k, d in zip(key, descr):
              if lbl:
                ax.plot(best['epoch'], best[k][0], marker="o",color="red")
                ax.plot(best['epoch'], best[k][1], marker="o",color="green")
              else:
                ax.plot(best['epoch'], best[k][0], marker="o",color="red", label=f"Best Model's SpearCorr for the Train Set")
                ax.plot(best['epoch'], best[k][1], marker="o",color="green", label=f"Best Model's SpearCorr for the Dev Set")
              lbl=True
          for f, d in zip(function, descr):
            ax.plot(epochs, f[0,:ep], label=f"Train Set for {d}")
            ax.plot(epochs, f[1,:ep], label=f"Dev Set for {d}")  
        else:
          ax.plot(epochs, function[0,:ep], label="Train Set")
          ax.plot(epochs, function[1,:ep], label="Dev Set")
    
          if best is not None:
              ax.plot(best['epoch'], best[key][0], marker="o",color="red", label=f"Best Model's {descr} for the Train Set")
              ax.plot(best['epoch'], best[key][1], marker="o",color="green", label=f"Best Model's {descr} for the Dev Set")
          
        ax.set_title(title)
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Values')
        ax.grid()
        ax.legend()
    
    plt.tight_layout()
    plt.show()

def plot_corr(y, y_pred, titles, multi=False):      

    n_tasks = 3 if multi else 1
    figure, axes = plt.subplots(nrows=1, ncols=n_tasks, figsize = (n_tasks*7 + 3,8))
    _ = [print(f'Spearman Correlation for {titles[i]}: {stats.spearmanr(y_pred[i], y[i])[0]}') for i in range(n_tasks)]
    y_cpu = [[y0.clone().detach().cpu() for y0 in y[i]] for i in range(n_tasks)]
    y_pred_cpu = [[y0.clone().detach().cpu() for y0 in y_pred[i]] for i in range(n_tasks)]

    if not multi:
      
      axes.scatter(y_cpu[0], y_pred_cpu[0])
      axes.set_title(titles[0])
      axes.set_xlabel('Ground Truth')
      axes.set_ylabel('Predictions')
      axes.grid()
    else:

      for ax, index, title in zip(axes.flatten(), range(n_tasks), titles):

        ax.scatter(y_cpu[index], y_pred_cpu[index])
        ax.set_title(title)
        ax.set_xlabel('Ground Truth')
        ax.set_ylabel('Predictions')
        ax.grid()

    plt.tight_layout()
    plt.show()    
  
def plot_stats(losses, accuracy, f1, recall, epochs, ep, best=None):
    figure, axes = plt.subplots(nrows=2, ncols=2, figsize = (10,8))
    titles = ['Loss', 'Accuracy Score', 'F1 Score', 'Recall Score']
    functions = [losses, accuracy, f1, recall]
    best_keys = ['loss', 'accuracy', 'f1', 'recall']
    descrs = ['Loss', 'Accuracy Score', 'F1 Score', 'Recall Score']
  
    for ax, title, function, key, descr in zip(axes.flatten(), titles, functions, best_keys, descrs):
        ax.plot(epochs, function[0,:ep], label="Train Set")
        ax.plot(epochs, function[1,:ep], label="Dev Set")
    
        if best is not None:
            ax.plot(best['epoch'], best[key][0], marker="o",color="red", label=f"Best Model's {descr} for the Train Set")
            ax.plot(best['epoch'], best[key][1], marker="o",color="green", label=f"Best Model's {descr} for the Dev Set")
          
        ax.set_title(title)
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Values')
        ax.grid()
        ax.legend()
  
    plt.tight_layout()
    plt.show()