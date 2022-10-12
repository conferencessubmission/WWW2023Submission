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
from models_ import OnlyClassfnTwoStage
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
taxonomy_path = './taxonomy_hierarchy'
model_type = 'roberta'

save_model_folder = 'Models_only_classfn_two_stage_rule_based_32'

if model_type == 'roberta':
  save_model_folder += '_roberta'
  path_dataset += '_roberta'

if os.path.exists(save_model_folder) == False:
  os.mkdir(save_model_folder)

embeddings1 = []
embeddings2 = []
# siamese_labels = []

total_shards = 10


shard_num = int(sys.argv[1])

pos_pairs_list = []
neg_pairs_list = []

with open(os.path.join('./', 'pos_pairs_rule_based.pickle'), 'rb') as f:
  init_pos_pairs_list = pickle.load(f)

num_samples = int(len(init_pos_pairs_list)/total_shards)

init_pos_pairs_list = init_pos_pairs_list[(shard_num-1)*num_samples:shard_num*num_samples]

pos_pairs_list.extend(init_pos_pairs_list)
pos_pairs_list.extend([(it[1], it[0]) for it in init_pos_pairs_list])

with open(os.path.join('./', 'neg_pairs_rule_based.pickle'), 'rb') as f:
  init_neg_pairs_list = pickle.load(f)
init_neg_pairs_list = init_neg_pairs_list[(shard_num-1)*num_samples:shard_num*num_samples]

neg_pairs_list.extend(init_neg_pairs_list)
neg_pairs_list.extend([(it[1], it[0]) for it in init_neg_pairs_list])

print(len(pos_pairs_list), len(neg_pairs_list))

def get_text(filename):
  with open(os.path.join('../EManuals_data/EManuals', filename), 'r') as f:
    text = f.read()

  return [i.strip() for i in text.split('\n') if i.strip()!='']

# labels = []
emanuals1 = []
emanuals2 = []

for i in range(len(pos_pairs_list)): # already contains bothe permutations
  # labels.append(1)
  emanuals1.append(get_text(pos_pairs_list[i][0]))
  emanuals2.append(get_text(pos_pairs_list[i][1]))

# for i in range(len(pos_pairs_list)):
#   labels.append(1)
#   emanuals1.append(get_text(pos_pairs_list[i][1])) # both permutations should go as inputs
#   emanuals2.append(get_text(pos_pairs_list[i][0])) # both permutations should go as inputs

for i in range(len(neg_pairs_list)): # already contains bothe permutations
  # labels.append(0)
  emanuals1.append(get_text(neg_pairs_list[i][0]))
  emanuals2.append(get_text(neg_pairs_list[i][1]))

# for i in range(len(neg_pairs_list)):
#   labels.append(0)
#   emanuals1.append(get_text(neg_pairs_list[i][1])) # both permutations should go as inputs
#   emanuals2.append(get_text(neg_pairs_list[i][0])) # both permutations should go as inputs

print(len(emanuals1), len(emanuals2))#, len(labels))

def truncate_or_pad(x, num):
  if len(x) >= num:
    return x[:num]
  else:
    return x + ['']*(num - len(x))

with open(os.path.join(taxonomy_path, 'taxonomy_hier_classes_labels.json'), 'r') as f:
  taxonomy_dict = json.load(f)

with open(os.path.join(taxonomy_path, 'filename_to_hier_class_labels.json'), 'r') as f:
  filename_to_labels = json.load(f)

num_levels = len(taxonomy_dict)

num_topics_each_level = []

for level in taxonomy_dict:
  num_topics_each_level.append(len(taxonomy_dict[level]))     # including 'None'

def create_heirarchical_data(num_levels, num_topics_each_level):
    
  heirarchical_encoding_sent1= []
  label_sent1 = []

# pos 1st item of the pair
    
  for i in range(len(pos_pairs_list)):
    level_label_dict = filename_to_labels[pos_pairs_list[i][0]]
    temp_list = []
    label_list = []    
    for j in range(num_levels):
      # for num_topic in num_topics_each_level:
      arr = list(np.zeros(num_topics_each_level[j]))
      index = level_label_dict[str(j)]
      arr[index] = 1
      temp_list.append(arr)
      label_list.append(index)

    heirarchical_encoding_sent1.append(temp_list)
    label_sent1.append(label_list)

  for i in range(len(neg_pairs_list)):
    level_label_dict = filename_to_labels[neg_pairs_list[i][0]]
    temp_list = []
    label_list = []    
    for j in range(num_levels):
      # for num_topic in num_topics_each_level:
      arr = list(np.zeros(num_topics_each_level[j]))
      index = level_label_dict[str(j)]
      arr[index] = 1
      temp_list.append(arr)
      label_list.append(index)

    heirarchical_encoding_sent1.append(temp_list)
    label_sent1.append(label_list)

  heirarchical_encoding_sent2= []
  label_sent2 = []

# pos 1st item of the pair
    
  for i in range(len(pos_pairs_list)):
    level_label_dict = filename_to_labels[pos_pairs_list[i][1]]
    temp_list = []
    label_list = []    
    for j in range(num_levels):
      # for num_topic in num_topics_each_level:
      arr = list(np.zeros(num_topics_each_level[j]))
      index = level_label_dict[str(j)]
      arr[index] = 1
      temp_list.append(arr)
      label_list.append(index)

    heirarchical_encoding_sent2.append(temp_list)
    label_sent2.append(label_list)

  for i in range(len(neg_pairs_list)):
    level_label_dict = filename_to_labels[neg_pairs_list[i][1]]
    temp_list = []
    label_list = []    
    for j in range(num_levels):
      # for num_topic in num_topics_each_level:
      arr = list(np.zeros(num_topics_each_level[j]))
      index = level_label_dict[str(j)]
      arr[index] = 1
      temp_list.append(arr)
      label_list.append(index)

    heirarchical_encoding_sent2.append(temp_list)
    label_sent2.append(label_list)

  return heirarchical_encoding_sent1, heirarchical_encoding_sent2, label_sent1, label_sent2

heirarchical_encoding_sent1, heirarchical_encoding_sent2, label_sent1, label_sent2 = create_heirarchical_data(num_levels, num_topics_each_level)

# heirarchical_encoding_sent2[1][0]

# label_sent1[:4]

def make_dataset(sents1, sents2, tokenizer, max_len_input, max_num_seqs, heirarchical_encoding_sent1, heirarchical_encoding_sent2, label_sent1, label_sent2):        

  all_input_ids1 = []
  all_input_ids2 = []
  all_attention_masks1 = []
  all_attention_masks2 = []
  if model_type!='roberta':
    all_tok_type_ids_1 = []
    all_tok_type_ids_2 = []
  # all_labels = []

  for sent1, sent2 in zip(sents1, sents2):

    encoded_input1 = tokenizer(truncate_or_pad(sent1, max_num_seqs), max_length = max_len_input, padding = 'max_length', truncation = True)
    encoded_input2 = tokenizer(truncate_or_pad(sent2, max_num_seqs), max_length = max_len_input, padding = 'max_length', truncation = True)

    # print(len(encoded_input1['input_ids']))
    # print(len(encoded_input2['input_ids']))

    all_input_ids1.append(encoded_input1['input_ids'])
    all_input_ids2.append(encoded_input2['input_ids'])

    all_attention_masks1.append(encoded_input1['attention_mask'])
    all_attention_masks2.append(encoded_input2['attention_mask'])
    if model_type!='roberta':
      all_tok_type_ids_1.append(encoded_input1['token_type_ids'])
      all_tok_type_ids_2.append(encoded_input2['token_type_ids'])

    # all_labels.append(int(label))

  all_input_ids1 = torch.as_tensor(all_input_ids1)
  all_input_ids2 = torch.as_tensor(all_input_ids2)

  print(len(all_input_ids1), len(all_input_ids2))

  all_attention_masks1 = torch.as_tensor(all_attention_masks1)
  all_attention_masks2 = torch.as_tensor(all_attention_masks2)
  if model_type!='roberta':
    all_tok_type_ids_1 = torch.as_tensor(all_tok_type_ids_1)
    all_tok_type_ids_2 = torch.as_tensor(all_tok_type_ids_2)

  # all_labels = torch.tensor(all_labels)

  heirarchical_1 = []
  heirarchical_2 = []
  heirarchical_3 = []
  heirarchical_4 = []
  heirarchical_5 = []
  heirarchical_6 = []
  heirarchical_7 = []  

  for i in range(len(heirarchical_encoding_sent1)):
    heirarchical_1.append(heirarchical_encoding_sent1[i][0])
    heirarchical_2.append(heirarchical_encoding_sent1[i][1])
    heirarchical_3.append(heirarchical_encoding_sent1[i][2])
    heirarchical_4.append(heirarchical_encoding_sent1[i][3])
    heirarchical_5.append(heirarchical_encoding_sent1[i][4])
    heirarchical_6.append(heirarchical_encoding_sent1[i][5])
    heirarchical_7.append(heirarchical_encoding_sent1[i][6])

  sent1_all_heirarchical_encodding_1 = torch.tensor( heirarchical_1).squeeze()
  sent1_all_heirarchical_encodding_2 = torch.tensor( heirarchical_2).squeeze()
  sent1_all_heirarchical_encodding_3 = torch.tensor( heirarchical_3).squeeze()
  sent1_all_heirarchical_encodding_4 = torch.tensor( heirarchical_4).squeeze()
  sent1_all_heirarchical_encodding_5 = torch.tensor( heirarchical_5).squeeze()
  sent1_all_heirarchical_encodding_6 = torch.tensor( heirarchical_6).squeeze()
  sent1_all_heirarchical_encodding_7 = torch.tensor( heirarchical_7).squeeze()  

  heirarchical_1 = []
  heirarchical_2 = []
  heirarchical_3 = []
  heirarchical_4 = []
  heirarchical_5 = []
  heirarchical_6 = []
  heirarchical_7 = []    

  for i in range(len(heirarchical_encoding_sent2)):
    heirarchical_1.append(heirarchical_encoding_sent2[i][0])
    heirarchical_2.append(heirarchical_encoding_sent2[i][1])
    heirarchical_3.append(heirarchical_encoding_sent2[i][2])
    heirarchical_4.append(heirarchical_encoding_sent2[i][3])
    heirarchical_5.append(heirarchical_encoding_sent2[i][4])
    heirarchical_6.append(heirarchical_encoding_sent2[i][5])
    heirarchical_7.append(heirarchical_encoding_sent2[i][6])

  sent2_all_heirarchical_encodding_1 = torch.tensor( heirarchical_1).squeeze()
  sent2_all_heirarchical_encodding_2 = torch.tensor( heirarchical_2).squeeze()
  sent2_all_heirarchical_encodding_3 = torch.tensor( heirarchical_3).squeeze()
  sent2_all_heirarchical_encodding_4 = torch.tensor( heirarchical_4).squeeze()
  sent2_all_heirarchical_encodding_5 = torch.tensor( heirarchical_5).squeeze()
  sent2_all_heirarchical_encodding_6 = torch.tensor( heirarchical_6).squeeze()
  sent2_all_heirarchical_encodding_7 = torch.tensor( heirarchical_7).squeeze()  

  #print(all_heirarchical_encodding_1.shape)
  #print(all_heirarchical_encodding_3.shape)

  all_labels_sent1 = torch.tensor(label_sent1)
  all_labels_sent2 = torch.tensor(label_sent2)
  if model_type == 'roberta':
    dataset = TensorDataset(all_input_ids1, all_input_ids2, all_attention_masks1, all_attention_masks2, sent1_all_heirarchical_encodding_1,  
                            sent1_all_heirarchical_encodding_2,  sent1_all_heirarchical_encodding_3,  sent1_all_heirarchical_encodding_4,  sent1_all_heirarchical_encodding_5,  sent1_all_heirarchical_encodding_6,  sent1_all_heirarchical_encodding_7,
                            sent2_all_heirarchical_encodding_1,  sent2_all_heirarchical_encodding_2,  sent2_all_heirarchical_encodding_3,  
                            sent2_all_heirarchical_encodding_4,  sent2_all_heirarchical_encodding_5, sent2_all_heirarchical_encodding_6,  sent2_all_heirarchical_encodding_7, all_labels_sent1, all_labels_sent2)  
  else:
    dataset = TensorDataset(all_input_ids1, all_input_ids2, all_attention_masks1, all_attention_masks2, all_tok_type_ids_1, all_tok_type_ids_2, sent1_all_heirarchical_encodding_1,  
                            sent1_all_heirarchical_encodding_2,  sent1_all_heirarchical_encodding_3,  sent1_all_heirarchical_encodding_4,  sent1_all_heirarchical_encodding_5,  sent1_all_heirarchical_encodding_6,  sent1_all_heirarchical_encodding_7,
                            sent2_all_heirarchical_encodding_1,  sent2_all_heirarchical_encodding_2,  sent2_all_heirarchical_encodding_3,  
                            sent2_all_heirarchical_encodding_4,  sent2_all_heirarchical_encodding_5, sent2_all_heirarchical_encodding_6,  sent2_all_heirarchical_encodding_7, all_labels_sent1, all_labels_sent2)

  return dataset

model_path = 'bert-base-uncased'

if model_type == 'roberta':
  model_path = 'roberta-base'

tokenizer = AutoTokenizer.from_pretrained(model_path)

seq_len = 128
max_num_seqs = 32

train_dataset = make_dataset(emanuals1, emanuals2, tokenizer, seq_len, max_num_seqs, heirarchical_encoding_sent1, heirarchical_encoding_sent2, label_sent1, label_sent2)

from transformers import AutoModel
from hier_utils import custom_loss_only_classfn

#batch_size = 1
EPOCHS = 1
BATCH_SIZE = 4

# batch accumulation parameter
accum_iter = 8

emb_len = 768
output_size = 768

model = nn.DataParallel(OnlyClassfnTwoStage(num_topics_each_level))

if model_type == 'roberta':
  model = nn.DataParallel(OnlyClassfnTwoStage(num_topics_each_level,model_path))  

if shard_num > 1:
  model.load_state_dict(torch.load(os.path.join(save_model_folder, 'only_classfn_twostage_epochs_1_shard_{}.pt'.format(shard_num-1))))
  print('Loaded previous model')

model.to(device);

train_dataloader = DataLoader(train_dataset,BATCH_SIZE,shuffle=True, num_workers=8)
# scores = []
loss_list = []

# criterion_main = nn.BCELoss()
criterion_heirarchy = nn.CrossEntropyLoss()

# weight_main = 1
weight1 = 1
weight2 = 1
weight3 = 1
weight4 = 1
weight5 = 1
weight6 = 1
weight7 = 1

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
      heirarchy_out1_emb1, heirarchy_out2_emb1, heirarchy_out3_emb1, heirarchy_out4_emb1, heirarchy_out5_emb1, heirarchy_out6_emb1, heirarchy_out7_emb1, heirarchy_out1_emb2, heirarchy_out2_emb2, heirarchy_out3_emb2, heirarchy_out4_emb2, heirarchy_out5_emb2, heirarchy_out6_emb2, heirarchy_out7_emb2 = model.forward(batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device))
    else:
      heirarchy_out1_emb1, heirarchy_out2_emb1, heirarchy_out3_emb1, heirarchy_out4_emb1, heirarchy_out5_emb1, heirarchy_out6_emb1, heirarchy_out7_emb1, heirarchy_out1_emb2, heirarchy_out2_emb2, heirarchy_out3_emb2, heirarchy_out4_emb2, heirarchy_out5_emb2, heirarchy_out6_emb2, heirarchy_out7_emb2 = model.forward(batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device), batch[4].to(device), batch[5].to(device))
    
    loss, cls_loss_1_str, cls_loss_2_str = custom_loss_only_classfn([heirarchy_out1_emb1.float().to(device), heirarchy_out2_emb1.float().to(device), 
                             heirarchy_out3_emb1.float().to(device), heirarchy_out4_emb1.float().to(device), heirarchy_out5_emb1.float().to(device), heirarchy_out6_emb1.float().to(device), heirarchy_out7_emb1.float().to(device)], [heirarchy_out1_emb2.float().to(device), 
                             heirarchy_out2_emb2.float().to(device), heirarchy_out3_emb2.float().to(device), heirarchy_out4_emb2.float().to(device), heirarchy_out5_emb2.float().to(device), heirarchy_out6_emb2.float().to(device), heirarchy_out7_emb2.float().to(device)],
                             batch[-2].to(device), batch[-1].to(device), criterion_heirarchy, float(weight1), float(weight2), float(weight3), float(weight4), float(weight5), float(weight6), float(weight7))
    loss = loss / accum_iter
    print('Epoch: {}, Step: {}'.format(epoch_i + 1, step + 1))
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
      torch.save(model.state_dict(), os.path.join(save_model_folder, 'only_classfn_twostage_epochs_{}_{}_{}_shard_{}.pt'.format(epoch_i + 1, step + 1, BATCH_SIZE, shard_num)))
  avg_train_loss = total_train_loss / len(train_dataloader)
  loss_list.append(avg_train_loss)
  print('Loss after epoch {} = {}'.format(epoch_i + 1, avg_train_loss))
torch.save(model.state_dict(), os.path.join(save_model_folder, 'only_classfn_twostage_epochs_{}_shard_{}.pt'.format(epoch_i + 1, shard_num)))
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

