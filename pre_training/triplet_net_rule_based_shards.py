# -*- coding: utf-8 -*-
'''
!pip install -U sentence-transformers -q

!pip install pandarallel -q

!pip install transformers -q
'''

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import random as rn
# import seaborn as sns
import ast
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, average_precision_score, precision_score, precision_recall_curve
import pickle
import nltk
import math
import os
import sys
import json
import random
import re
# from pandarallel import pandarallel
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import (DataLoader, RandomSampler, WeightedRandomSampler, SequentialSampler, TensorDataset)
import warnings
import sys
import time
# warnings.filterwarnings('ignore')
# pandarallel.initialize(progress_bar = True)
from tqdm import tqdm
from models_ import TripletNetwork

start_time = time.time()

SEED = 0
rn.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
device = 'cuda'

path_dataset = './triplet_rule_based_sentemb_shards' # contains embeddings
# taxonomy_path = './taxonomy_hierarchy'
save_model_folder = 'Models_rule_based_triplet'

model_type = 'roberta'

if model_type == 'roberta':
  save_model_folder += '_roberta'
  path_dataset += '_roberta'

if os.path.exists(save_model_folder) == False:
  os.mkdir(save_model_folder)

'''
with open(path_dataset + 'embedding1', 'rb') as f:
  embeddings_1 = pickle.load(f)
  f.close()

with open(path_dataset + 'embedding2', 'rb') as f:
  embeddings_2 = pickle.load(f)
  f.close()

df = pd.read_csv(path_dataset + 'dummy_data.csv')

display(df)

len(embeddings_1), embeddings_1[0].shape
'''
'''
with open(os.path.join(path_dataset, 'sentence_embeddings_dict_1_5.pickle'), 'rb') as f:
  embs_dict_1 = pickle.load(f)
print('1st done')
with open(os.path.join(path_dataset, 'sentence_embeddings_dict_2_5.pickle'), 'rb') as f:
  embs_dict_2 = pickle.load(f)
print('2nd done')
with open(os.path.join(path_dataset, 'sentence_embeddings_dict_3_5.pickle'), 'rb') as f:
  embs_dict_3 = pickle.load(f)
print('3rd done')
with open(os.path.join(path_dataset, 'sentence_embeddings_dict_4_5.pickle'), 'rb') as f:
  embs_dict_4 = pickle.load(f)
print('4th done')
with open(os.path.join(path_dataset, 'sentence_embeddings_dict_5_5.pickle'), 'rb') as f:
  embs_dict_5 = pickle.load(f)
print('5th done')
# with open(os.path.join(path_dataset, 'sentence_embeddings_dict_6.pickle'), 'rb') as f:
#   embs_dict_6 = pickle.load(f)
# print('6th done')

embs_dict = {**embs_dict_1, **embs_dict_2, **embs_dict_3, **embs_dict_4, **embs_dict_5}

print('Loaded embeddings')
print()

with open(os.path.join(path_dataset, 'pos_pairs.pickle'), 'rb') as f:
  pos_pairs_list = pickle.load(f)

with open(os.path.join(path_dataset, 'neg_pairs.pickle'), 'rb') as f:
  neg_pairs_list = pickle.load(f)

labels = []
embeddings1 = []
embeddings2 = []

for i in range(len(pos_pairs_list)):
  labels.append(1)
  embeddings1.append(embs_dict[pos_pairs_list[i][0]])
  embeddings2.append(embs_dict[pos_pairs_list[i][1]])

for i in range(len(pos_pairs_list)):
  labels.append(1)
  embeddings1.append(embs_dict[pos_pairs_list[i][1]]) # both permutations should go as inputs
  embeddings2.append(embs_dict[pos_pairs_list[i][0]]) # both permutations should go as inputs

for i in range(len(neg_pairs_list)):
  labels.append(0)
  embeddings1.append(embs_dict[neg_pairs_list[i][0]])
  embeddings2.append(embs_dict[neg_pairs_list[i][1]])

for i in range(len(neg_pairs_list)):
  labels.append(0)
  embeddings1.append(embs_dict[neg_pairs_list[i][1]]) # both permutations should go as inputs
  embeddings2.append(embs_dict[neg_pairs_list[i][0]]) # both permutations should go as inputs
'''

# embeddings = []
# embeddings_pos = []
# embeddings_neg = []
# labels = []


total_shards = 10


shard_num = int(sys.argv[1])

with open(os.path.join(path_dataset, 'anchor_embs_{}_{}.pickle'.format(shard_num, total_shards)), 'rb') as f:
  embeddings = pickle.load(f)

with open(os.path.join(path_dataset, 'pos_embs_{}_{}.pickle'.format(shard_num, total_shards)), 'rb') as f:
  embeddings_pos = pickle.load(f)

with open(os.path.join(path_dataset, 'neg_embs_{}_{}.pickle'.format(shard_num, total_shards)), 'rb') as f:
  embeddings_neg = pickle.load(f)

# embeddings.extend(tmp_anchor)
# embeddings_pos.extend(tmp_pos)
# embeddings_neg.extend(tmp_neg)

print(len(embeddings), len(embeddings_pos), len(embeddings_neg))

def make_dataset(embeddings, embeddings_pos, embeddings_neg):        

        #embeddin
  all_emanuals = torch.tensor(embeddings).squeeze()
  print(all_emanuals.shape)
  all_emanuals_pos = torch.tensor(embeddings_pos).squeeze()
  print(all_emanuals_pos.shape)
  all_emanuals_neg = torch.tensor(embeddings_neg).squeeze()
  print(all_emanuals_neg.shape)
  dataset = TensorDataset(all_emanuals, all_emanuals_pos, all_emanuals_neg)

  return dataset

train_dataset = make_dataset(embeddings, embeddings_pos, embeddings_neg)

len(train_dataset)

print()
print('Training dataset created')
print()

from transformers import AutoModel

model_path = 'bert-base-uncased'

if model_type == 'roberta':
  model_path = 'roberta-base'

# below is simple siamese without attention


# combining bahadanu and siamese for setting one
# working for BS 1
'''class SiameseNetwork(nn.Module):

        def __init__(self, hidden_size = 768, output_size = 768, n_layers=1, drop_prob=0.1):
                super(SiameseNetwork, self).__init__()

                self.hidden_size = hidden_size
                self.output_size = output_size
                self.n_layers = n_layers
                self.drop_prob = drop_prob

                
                self.fc_hidden = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
                self.fc_encoder = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
                self.weight = nn.Parameter(torch.FloatTensor(1, hidden_size))
                self.classifier = nn.Linear(self.hidden_size, self.output_size)


                self.model_layer = AutoModel.from_pretrained(model_path)
                self.dense_layer = nn.Linear(768, 1)
                self.sigmoid = nn.Sigmoid()

        def forward(self, embedding1, embedding2):

                output1 = self.model_layer.forward(inputs_embeds = embedding1)[0]
                output2 = self.model_layer.forward(inputs_embeds = embedding2)[0]

                encoder_outputs = output2.squeeze()
                
                # Calculating Alignment Scores
                x = torch.tanh(self.fc_hidden(output1)+self.fc_encoder(encoder_outputs))
                
                alignment_scores = x.bmm(self.weight.unsqueeze(2)) 
                
                # Softmaxing alignment scores to get Attention weights
                attn_weights = F.softmax(alignment_scores.view(1,-1), dim=1)
                

                # Multiplying the Attention weights with encoder outputs to get the context vector
                context_vector = torch.bmm(attn_weights.unsqueeze(0),
                                        encoder_outputs.unsqueeze(0))
                

                output = self.classifier(context_vector)
    
                #avg_output_1 = torch.mean(output1, 1, True)
                #avg_output_2 = torch.mean(output2, 1, True)
                #diff = torch.sub(avg_output_1, avg_output_2, alpha = 1)
                
                output = self.dense_layer(output)
                similarity = self.sigmoid(output)

                return similarity'''

# combining bahadanu and siamese for setting one first bert then fed to Bahdanu
'''
class SiameseNetwork(nn.Module):

        def __init__(self, hidden_size = 768, output_size = 768, n_layers=1, drop_prob=0.1):
                super(SiameseNetwork, self).__init__()

                self.hidden_size = hidden_size
                self.output_size = output_size
                self.n_layers = n_layers
                self.drop_prob = drop_prob

                
                self.fc_hidden = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
                self.fc_encoder = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
                self.weight = nn.Parameter(torch.FloatTensor(1, hidden_size))
                self.classifier = nn.Linear(self.hidden_size, self.output_size)


                self.model_layer = AutoModel.from_pretrained(model_path)
                self.dense_layer = nn.Linear(768, 1)
                
                self.dense_layer_1 = nn.Linear(768, 1)
                self.sigmoid = nn.Sigmoid()

        def forward(self, embedding1, embedding2):

                output1 = self.model_layer.forward(inputs_embeds = embedding1)[0]
                output2 = self.model_layer.forward(inputs_embeds = embedding2)[0]

                encoder_outputs_1 = output1
                encoder_outputs_2 = output2
                
                
                # Calculating Alignment Scores
                x_1 = torch.tanh(self.fc_encoder(encoder_outputs_1))
                x_2 = torch.tanh(self.fc_encoder(encoder_outputs_2))

                #print(x_1.shape)
               # print(self.weight.unsqueeze(2).shape)

                #alignment_scores_1 = x_1.bmm(self.weight.unsqueeze(2))
                #alignment_scores_2 = x_2.bmm(self.weight.unsqueeze(2))

                alignment_scores_1 = self.dense_layer_1(x_1)
                alignment_scores_2 = self.dense_layer_1(x_2)


                #print(alignment_scores_1.shape) 
                
                # Softmaxing alignment scores to get Attention weights
                attn_weights_1 = F.softmax(alignment_scores_1, dim=1)
                attn_weights_2 = F.softmax(alignment_scores_2, dim=1)
                
                #print()
                #print(attn_weights_1.shape)
                #print(attn_weights_1.unsqueeze(0).shape)

                #print( encoder_outputs_1.shape)
                #print( encoder_outputs_1.unsqueeze(0).shape)
                

                # Multiplying the Attention weights with encoder outputs to get the context vector
                #context_vector_1 = self.dense_layer_1(encoder_outputs_1)
                
                context_vector_1 = torch.bmm(attn_weights_1.view(BATCH_SIZE,1,4),
                                        encoder_outputs_1)
                
                context_vector_2 = torch.bmm(attn_weights_2.view(BATCH_SIZE,1,4),
                                        encoder_outputs_2)
                

                #output = self.classifier(context_vector)
    
                #avg_output_1 = torch.mean(output1, 1, True)
                #avg_output_2 = torch.mean(output2, 1, True)

                #print()
                #print(context_vector_1.shape)
                #print()
                #print(context_vector_2.shape)
                #print()
                avg_context = torch.mean(torch.add(context_vector_1, context_vector_2),1, True)
                #print()

                #print(diff.shape)

                output = self.dense_layer(avg_context)
                #print(output.shape)

                similarity = self.sigmoid(output)
                #print()
                #print(similarity.shape)

                return similarity
'''

#batch_size = 1
EPOCHS = 1
BATCH_SIZE = 32

# seq_len = 64 # num of sentences per emanual
emb_len = 768
output_size = 768

model = nn.DataParallel(TripletNetwork(model_path))

if shard_num > 1:
  model.load_state_dict(torch.load(os.path.join(save_model_folder, 'bert_triplet_epochs_1_shard_{}.pt'.format(shard_num-1))))
  print('Loaded previous model')

model.to(device);



train_dataloader = DataLoader(train_dataset,BATCH_SIZE,shuffle=True, num_workers=8)

print()
print(time.time() - start_time)
print()
print('Loaded all data')
print()
print()

scores = []
loss_list = []
criterion = nn.TripletMarginLoss(margin=1.0, p=2)
optimizer = AdamW(model.parameters(),
                  lr = 5e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )
total_steps = len(train_dataloader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)

for epoch_i in tqdm(range(EPOCHS)):
  total_train_loss = 0
  model.train()
  epoch_iterator = tqdm(train_dataloader, desc="Iteration")

  
  for step, batch in enumerate(epoch_iterator):
    model.zero_grad()
    anchor_out, pos_out, neg_out = model.forward(batch[0].to(device), batch[1].to(device), batch[2].to(device))
    
    loss = criterion(anchor_out.to(device), pos_out.to(device), neg_out.to(device))
    total_train_loss += loss.item()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    if (epoch_i * len(train_dataloader) + step + 1) % 400 == 0:
      torch.save(model.state_dict(), os.path.join(save_model_folder, 'bert_triplet_epochs_{}_{}_{}_shard_{}.pt'.format(epoch_i + 1, step + 1, BATCH_SIZE, shard_num)))
  avg_train_loss = total_train_loss / len(train_dataloader)
  loss_list.append(avg_train_loss)
  print('Loss after epoch {} = {}'.format(epoch_i + 1, avg_train_loss))

torch.save(model.state_dict(), os.path.join(save_model_folder, 'bert_triplet_epochs_{}_shard_{}.pt'.format(epoch_i + 1, shard_num)))

print(loss_list)
# for step, batch in tqdm(enumerate(train_dataloader)):

#         score = model.forward(batch[0].to(device), batch[1].to(device))
#         scores.append(score)

# print(float(scores[2][0][0][0]))

# i = 0
# actual_label = []
# pred_label = []
# for step, batch in tqdm(enumerate(train_dataloader)):

#         actual_label.append(batch[2])
#         if float(scores[i][0][0][0]) > 0.5:
#             pred_label.append(1)
#         else:
#             pred_label.append(0)       

#         i = i + 1

# from sklearn.metrics import accuracy_score

# acc = accuracy_score(pred_label, actual_label)
# print(acc)

