from spotlight.cross_validation import random_train_test_split
from spotlight.datasets.movielens import get_movielens_dataset
from spotlight.evaluation import rmse_score, precision_recall_score, mrr_score
from spotlight.factorization.explicit import ExplicitFactorizationModel
from spotlight.factorization.implicit import ImplicitFactorizationModel
from spotlight.factorization.representations import BilinearNet
from spotlight.sequence.implicit import ImplicitSequenceModel
from spotlight.sequence.representations import CNNNet, LSTMNet
from spotlight.evaluation import sequence_mrr_score
from spotlight.interactions import Interactions
import torch
import pandas as pd
import numpy as np
import pickle
from collections import Counter
import os

from arg_parser import recommend_train

args = recommend_train()
model_mode = args.model_mode
train_path = args.train_path
test_path = args.test_path
save_file = os.path.join("recommend_model", args.save_file)
cleaning_train = args.cleaning_train
cleaning_test = args.cleaning_test
n_iter = args.n_iter

data_train = pd.read_excel(train_path,index_col=0, converters = {"id":np.int32})
data_test = pd.read_excel(test_path,index_col=0, converters = {"id":np.int32})
data_all = pd.concat([data_train,data_test])

if cleaning_train:
    data_train = data_train[data_train["fault"] == 1]
if cleaning_test:
    data_test = data_test[data_test["fault"] == 1]

foods_items = max(np.array(list(map(np.int32,data_all["menu_id"]))))+1
num_user = max(np.array(list(map(np.int32,data_all["id"]))))+100
for data,name in zip((data_train,data_test),("train","test")):
  data["time"] = pd.to_datetime(data["time"])
  data = data.sort_values(by=['time'], axis=0)
  timeStamps = np.arange(len(data))
  ids = np.array(list(map(np.int32,data["id"])))
  foods = np.array(list(map(np.int32,data["menu_id"])))
  ratings = np.array(list(map(np.float32,data["rating"])))
  dataset = Interactions(user_ids=ids,item_ids=foods,ratings=ratings,num_users=int(num_user),num_items=int(foods_items),timestamps=timeStamps)
  
  if name == "test":
    dataset_test = dataset
  elif name == "train":
    dataset_train = dataset
    
if model_mode.lower() == "ifm":
    model = ImplicitFactorizationModel(n_iter=n_iter)
if model_mode.lower() == "efm":
    model = ExplicitFactorizationModel(n_iter=n_iter)
if model_mode.lower() == "cnn":
    net = CNNNet(num_items=int(foods_items))
    model = ImplicitSequenceModel(n_iter=n_iter, use_cuda=torch.cuda.is_available(), representation=net)
    
model.fit(dataset_train)

with open(save_file, 'wb') as f:
    pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)

if model_mode.lower() == "cnn":
    mrr = sequence_mrr_score(model, dataset_test)
else:
    mrr = mrr_score(model, dataset_test)
    
print("mrr = ",len(mrr))
print("mean mrr = ",sum(mrr)/len(mrr))
rank = 1/(sum(mrr)/len(mrr))
print("average rank = ",rank)