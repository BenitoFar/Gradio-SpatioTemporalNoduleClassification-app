from functools import partial
import random
import os
import numpy as np
import optuna
from optuna.samplers import TPESampler
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import cross_val_score, StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc, confusion_matrix
from sklearn.calibration import calibration_curve
import seaborn as sns
from torch.nn.utils import prune

#create GRU class from scratch: do not use torch GRU implementation 
class GRUScratch(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(GRUScratch, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.W_z = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_r = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_h = nn.Linear(input_size + hidden_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
    
    def forward(self, x, h0, mask=None):
        h = h0
        outputs = []
        for i in range(x.size(1)):
            x_t = x[:, i, :]
            z_t = self.sigmoid(self.W_z(torch.cat((x_t, h.squeeze(0)), dim=1)))
            if mask is not None:
                z_t = torch.where(mask[:, i].unsqueeze(1), torch.ones_like(z_t), z_t)
            r_t = self.sigmoid(self.W_r(torch.cat((x_t, h.squeeze(0)), dim=1)))
            h_t = z_t * h + (1 - z_t) * self.tanh(self.W_h(torch.cat((x_t, r_t * h.squeeze(0)), dim=1)))
            h = h_t
            outputs.append(h.unsqueeze(2))
        outputs = torch.cat(outputs, dim=2).squeeze(0)
        return outputs, h
    
class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_rnn_layers, drop_prob, hidden_size_att, num_classes, device, attention_type='multiheadatt', temporal_dropout = False, num_heads=4):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.hidden_size_att = hidden_size_att
        self.num_rnn_layers = num_rnn_layers
        self.attention_type = attention_type
        self.device = device
        self.temporal_dropout = temporal_dropout
        self.drop_prob = drop_prob
        # self.rnn = nn.GRU(input_size, hidden_size, num_rnn_layers, batch_first=True, dropout=drop_prob)
        self.rnn = GRUScratch(input_size, hidden_size, num_rnn_layers, drop_prob)
        # self.layer_norm = nn.LayerNorm(hidden_size)
        
        if attention_type == 'multiheadatt':
            self.attention = nn.MultiheadAttention(hidden_size, num_heads, dropout=drop_prob)
            self.dropout = nn.Dropout(drop_prob)
            self.fc = nn.Linear(hidden_size, num_classes)
        elif self.attention_type == 'globalatt':
            #create a function that performs the operation score(h_t, H) = h_t^T W_a H
            self.W_a = nn.Linear(hidden_size, hidden_size)
            self.W_c = nn.Linear(2*hidden_size, hidden_size_att)
            self.dropout = nn.Dropout(drop_prob)
            self.fc = nn.Linear(hidden_size_att, num_classes)
        elif self.attention_type == 'none':
            self.dropout = nn.Dropout(drop_prob)
            self.fc = nn.Linear(hidden_size, num_classes)
        else:
            raise ValueError("Unsupported attention type. Choose from 'multiheadatt', 'globalatt'.")

        self.sigmoid = nn.Sigmoid()
        
    def freeze_layer(self, layer):
        for param in layer.parameters():
            param.requires_grad = False

    def forward(self, x):
        #get mask: where x == -1 (constant value)
        mask = (x == -1).any(dim=-1)
        
        mask_mod = mask
        #add dropout to axis = 1 (time steps): TODO
        if self.temporal_dropout:
            if self.training:
                for batch_idx in range(x.size(0)):
                    if np.array_equal(mask_mod[batch_idx].cpu().numpy(), [False, False, False]):
                        possibilities = [[False, False, False], [False, False, True], [False, True, False], [True, False, False], [False, True, True], [True, False, True], [True, True, False]]
                        #choose a random possibility
                        probabilities = [0.5] + [0.5 / (len(possibilities) - 1)] * (len(possibilities) - 1)
                        mask_mod[batch_idx] = torch.tensor(random.choices(possibilities, probabilities)[0]).to(self.device)
                    elif np.array_equal(mask_mod[batch_idx].cpu().numpy(), [False, False, True]):
                        possibilities = [[False, False, True], [True, False, True], [False, True, True]]
                        probabilities = [0.5] + [0.5 / (len(possibilities) - 1)] * (len(possibilities) - 1)
                        mask_mod[batch_idx] = torch.tensor(random.choices(possibilities, probabilities)[0]).to(self.device)
                    elif np.array_equal(mask_mod[batch_idx].cpu().numpy(), [False, True, False]):
                        possibilities = [[False, True, False], [True, True, False], [False, True, True]]
                        probabilities = [0.5] + [0.5 / (len(possibilities) - 1)] * (len(possibilities) - 1)
                        mask_mod[batch_idx] = torch.tensor(random.choices(possibilities, probabilities)[0]).to(self.device)
                    elif np.array_equal(mask_mod[batch_idx].cpu().numpy(), [True, False, False]):
                        possibilities = [[True, False, False], [True, False, True], [True, True, False]]
                        probabilities = [0.5] + [0.5 / (len(possibilities) - 1)] * (len(possibilities) - 1)
                        mask_mod[batch_idx] = torch.tensor(random.choices(possibilities, probabilities)[0]).to(self.device)
                    else:
                        mask_mod[batch_idx] = mask_mod[batch_idx]
                        
                    ### NOTE: error when in the last batch (3, 3, 4096), not all time points are present at least once
                    # Determine the number of time points to remove based on the number of missing time points
                    # n_time_to_remove = max(0, 2 - mask_mod[batch_idx].sum().item())
                    # drop_prob = 0.5 if n_time_to_remove == 2 else 0.3 if n_time_to_remove == 1 else 0

                    # if n_time_to_remove > 0:
                    #     times_not_missing = torch.where(~mask_mod[batch_idx])[0]
                    #     potential_time_points_to_remove = random.sample(times_not_missing.tolist(), n_time_to_remove)
                    #     for time_point_to_remove in potential_time_points_to_remove:
                    #         if random.random() < drop_prob:
                    #             mask_mod[batch_idx, time_point_to_remove] = True
                    #             drop_prob -= 0.2

                if (mask_mod.sum(dim=0) == mask_mod.shape[0]).any():
                    mask = mask
                    # raise ValueError("All time points are missing in at least one batch")
                else:
                    mask = mask_mod
            
                # Modify x based on the updated mask
                x[mask] = -1
        
        #if mask is boolean and mask.shape = [N, T], check if at least there is a False for each column; if not raise error or assign mask = None
        if (mask.sum(dim=0) == mask.shape[0]).any():
            mask = None

        h0 = torch.zeros(self.num_rnn_layers, x.size(0), self.hidden_size) #.to(self.device)
        r_output, _ = self.rnn(x, h0, mask)

        #add layer normalization
        # r_output = self.layer_norm(r_output)
        
        if self.attention_type == 'multiheadatt':
            attn_output, attn_weights = self.attention(r_output, r_output, r_output, key_padding_mask = mask.permute(1, 0) if mask is not None else None)
            #get only the last output
            attn_output = attn_output[:,-1:,]
            attn_weights = None
        elif self.attention_type == 'globalatt':
            #score(h_t, H) = h_t^T W_a H
            score = torch.bmm(self.W_a(r_output), r_output[:,-1:,].transpose(1, 2))
            if mask is not None:
                score = score.masked_fill(mask.unsqueeze(2), float('-inf')) #mask.unsqueeze(2) == 1
            attn_weights = torch.softmax(score, dim=1)
            context_vector = torch.bmm(attn_weights.transpose(1, 2), r_output)
            #apply a non-linear function to the output of the attention mechanism: tanh(W_c [h_t, c_t])
            attn_output = torch.tanh(self.W_c(torch.cat((r_output[:,-1:,], context_vector), dim=-1)))
        elif self.attention_type == 'none':
            attn_output = r_output[:,-1:,]
            attn_weights = None
        else:
            raise ValueError("Unsupported attention type. Choose from 'multiheadatt',  or 'globalatt' or 'none' .")
        
        # if mask is not None:
        #     attn_weights = attn_weights * mask.unsqueeze(1).unsqueeze(2)
        #     attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-9)
            
        out = self.dropout(attn_output)
        out = self.fc(out)
        
        return self.sigmoid(out), attn_weights