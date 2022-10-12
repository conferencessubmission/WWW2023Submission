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
# import ast
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
from models_ import TwoStageTriplet
# import warnings
# warnings.filterwarnings('ignore')
# pandarallel.initialize(progress_bar = True)
from tqdm import tqdm

SEED = 0
rn.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
device = 'cuda'

path_dataset = './' # contains embeddings
# taxonomy_path = './taxonomy_hierarchy'

model_type = 'roberta'

save_model_folder = 'Models_two_stage_triplet_rule_based_32'

if model_type == 'roberta':
  save_model_folder += '_roberta'
  path_dataset += '_roberta'

if os.path.exists(save_model_folder) == False:
  os.mkdir(save_model_folder)

# embeddings1 = []
# embeddings2 = []
# siamese_labels = []

total_shards = 10


shard_num = int(sys.argv[1])

pos_pairs_list = []
neg_pairs_list = []

with open(os.path.join('./', 'triplets_rule_based.pickle'), 'rb') as f:
  triplets_list = pickle.load(f)

num_samples = int(len(triplets_list)/total_shards)

triplets_list = triplets_list[(shard_num-1)*num_samples:shard_num*num_samples]

print(len(triplets_list))

def get_text(filename):
  with open(os.path.join('../EManuals_data/EManuals', filename), 'r') as f:
    text = f.read()

  return [i.strip() for i in text.split('\n') if i.strip()!='']

emanuals1 = []
emanuals2 = []
emanuals3 = []

for i in range(len(triplets_list)): # already contains bothe permutations
  emanuals1.append(get_text(triplets_list[i][0]))
  emanuals2.append(get_text(triplets_list[i][1]))
  emanuals3.append(get_text(triplets_list[i][2]))

print(len(emanuals1), len(emanuals2), len(emanuals3))

def truncate_or_pad(x, num):
  if len(x) >= num:
    return x[:num]
  else:
    return x + ['']*(num - len(x))

def make_dataset(sents1, sents2, sents3, tokenizer, max_len_input, max_num_seqs):        

  all_input_ids1 = []
  all_input_ids2 = []
  all_input_ids3 = []
  all_attention_masks1 = []
  all_attention_masks2 = []
  all_attention_masks3 = []
  if model_type!='roberta':
    all_tok_type_ids_1 = []
    all_tok_type_ids_2 = []
    all_tok_type_ids_3 = []

  for sent1, sent2, sent3 in zip(sents1, sents2, sents3):

    encoded_input1 = tokenizer(truncate_or_pad(sent1, max_num_seqs), max_length = max_len_input, padding = 'max_length', truncation = True)
    encoded_input2 = tokenizer(truncate_or_pad(sent2, max_num_seqs), max_length = max_len_input, padding = 'max_length', truncation = True)
    encoded_input3 = tokenizer(truncate_or_pad(sent3, max_num_seqs), max_length = max_len_input, padding = 'max_length', truncation = True)

    # print(len(encoded_input1['input_ids']))
    # print(len(encoded_input2['input_ids']))

    all_input_ids1.append(encoded_input1['input_ids'])
    all_input_ids2.append(encoded_input2['input_ids'])
    all_input_ids3.append(encoded_input3['input_ids'])

    all_attention_masks1.append(encoded_input1['attention_mask'])
    all_attention_masks2.append(encoded_input2['attention_mask'])
    all_attention_masks3.append(encoded_input3['attention_mask'])
    if model_type!='roberta':
      all_tok_type_ids_1.append(encoded_input1['token_type_ids'])
      all_tok_type_ids_2.append(encoded_input2['token_type_ids'])
      all_tok_type_ids_3.append(encoded_input3['token_type_ids'])

  all_input_ids1 = torch.as_tensor(all_input_ids1)
  all_input_ids2 = torch.as_tensor(all_input_ids2)
  all_input_ids3 = torch.as_tensor(all_input_ids3)  

  print(len(all_input_ids1), len(all_input_ids2), len(all_input_ids3))

  all_attention_masks1 = torch.as_tensor(all_attention_masks1)
  all_attention_masks2 = torch.as_tensor(all_attention_masks2)
  all_attention_masks3 = torch.as_tensor(all_attention_masks3)  
  if model_type!='roberta':
    all_tok_type_ids_1 = torch.as_tensor(all_tok_type_ids_1)
    all_tok_type_ids_2 = torch.as_tensor(all_tok_type_ids_2)
    all_tok_type_ids_3 = torch.as_tensor(all_tok_type_ids_3)  

  if model_type == 'roberta':
    dataset = TensorDataset(all_input_ids1, all_input_ids2, all_input_ids3, all_attention_masks1, all_attention_masks2, all_attention_masks3)
  else:
    dataset = TensorDataset(all_input_ids1, all_input_ids2, all_input_ids3, all_attention_masks1, all_attention_masks2, all_attention_masks3, all_tok_type_ids_1, all_tok_type_ids_2, all_tok_type_ids_3)

  return dataset

model_path = 'huawei-noah/TinyBERT_General_6L_768D'

if model_type == 'roberta':
  # model_path = 'roberta-base'
  model_path = 'distilroberta-base'

tokenizer = AutoTokenizer.from_pretrained(model_path)

seq_len = 128
max_num_seqs = 32

train_dataset = make_dataset(emanuals1, emanuals2, emanuals3, tokenizer, seq_len, max_num_seqs)

from transformers import AutoModel
from hier_utils import custom_loss_triplet

#batch_size = 1
EPOCHS = 1
BATCH_SIZE = 4

# batch accumulation parameter
accum_iter = 8

emb_len = 768
output_size = 768

model = nn.DataParallel(TwoStageTriplet())

if model_type == 'roberta':
  model = nn.DataParallel(TwoStageTriplet(model_path))  

if shard_num > 1:
  model.load_state_dict(torch.load(os.path.join(save_model_folder, 'twostagetriplet_epochs_1_shard_{}.pt'.format(shard_num-1))))
  print('Loaded previous model')

model.to(device);

train_dataloader = DataLoader(train_dataset,BATCH_SIZE,shuffle=True, num_workers=8)
scores = []
loss_list = []

criterion = nn.TripletMarginLoss(margin=1.0, p=2)

# weight_main = 1
# weight1 = 1
# weight2 = 1
# weight3 = 1
# weight4 = 1
# weight5 = 1
# weight6 = 1
# weight7 = 1

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

  model.zero_grad()

  
  for step, batch in enumerate(epoch_iterator):
    # model.zero_grad()
    if model_type == 'roberta':
      anchor_out, pos_out, neg_out = model.forward(batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device), batch[4].to(device), batch[5].to(device))
    else:
      anchor_out, pos_out, neg_out = model.forward(batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device), batch[4].to(device), batch[5].to(device), batch[6].to(device), batch[7].to(device), batch[8].to(device))
    
    loss = criterion(anchor_out, pos_out, neg_out)
    loss = loss / accum_iter
    # print('Epoch: {}, Step: {}'.format(epoch_i + 1, step + 1))
    total_train_loss += loss.item()
    # print('Total Loss: {}, Siamese Loss: {}, cls losses 1: {}, cls losses 2: {}'.format(loss, siamese_loss, cls_loss_1_str, cls_loss_2_str))
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if ((step + 1) % accum_iter == 0) or (step + 1 == len(train_dataloader)):

      optimizer.step()
      model.zero_grad()
      # optimizer.zero_grad()

      scheduler.step()
    if (epoch_i * len(train_dataloader) + step + 1) % 10000 == 0:
      torch.save(model.state_dict(), os.path.join(save_model_folder, 'twostagetriplet_epochs_{}_{}_{}_shard_{}.pt'.format(epoch_i + 1, step + 1, BATCH_SIZE, shard_num)))
  avg_train_loss = total_train_loss / len(train_dataloader)
  loss_list.append(avg_train_loss)
  print('Loss after epoch {} = {}'.format(epoch_i + 1, avg_train_loss))
torch.save(model.state_dict(), os.path.join(save_model_folder, 'twostagetriplet_epochs_{}_shard_{}.pt'.format(epoch_i + 1, shard_num)))
print(loss_list)

# for epoch_i in tqdm(range(EPOCHS)):
#   epoch_iterator = tqdm(train_dataloader, desc="Iteration")
  
#   for step, batch in enumerate(epoch_iterator):
    
#     score, heirarchy_out1_emb1, heirarchy_out2_emb1, heirarchy_out3_emb1, heirarchy_out4_emb1, heirarchy_out5_emb1, heirarchy_out1_emb2, heirarchy_out2_emb2, heirarchy_out3_emb2, heirarchy_out4_emb2, heirarchy_out5_emb2 = model.forward(batch[0].to(device), batch[1].to(device))
    #print(batch[-1])
    #print(heirarchy_out1_emb1.shape)
    # loss = model.custom_loss(score.reshape(BATCH_SIZE), batch[2].float().to(device),   [heirarchy_out1_emb1.float().to(device), heirarchy_out2_emb1.float().to(device), 
    #                          heirarchy_out3_emb1.float().to(device), heirarchy_out4_emb1.float().to(device), heirarchy_out5_emb1.float().to(device)], [heirarchy_out1_emb2.float().to(device), 
    #                          heirarchy_out2_emb2.float().to(device), heirarchy_out3_emb2.float().to(device), heirarchy_out4_emb2.float().to(device), heirarchy_out5_emb2.float().to(device)],
    #                          batch[-2].to(device), batch[-1].to(device), criterion_main, criterion_heirarchy, float(weight_main), float(weight1), float(weight2), float(weight3), float(weight4), float(weight5))
    
    # loss.backward()

