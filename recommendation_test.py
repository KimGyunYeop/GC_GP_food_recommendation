from spotlight.interactions import Interactions
import pandas as pd
import numpy as np
import pickle
import argparse
import json

mode = "ifm"

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--id', required=True, type=str,  help='an integer for the accumulator')
args = parser.parse_args()

data = pd.read_csv("test_data_6.csv",index_col=0,encoding='CP949')

data = data.loc[data["id"] == args.id]
data.loc[:,"id"] = 214
ids = np.array(list(map(np.int32,data["id"])))
foods = np.array(list(map(np.int32,data["menu_id"])))
ratings = np.array(list(map(np.float32,data["rating"])))
dataset = Interactions(user_ids=ids,item_ids=foods,num_users=215, ratings=ratings)
print(max(foods))
with open('train_model_ifm.pickle', 'rb') as f:
    model = pickle.load(f)
    
model.fit(dataset)

with open("result/ifm_result_6.json", "w") as json_file:
    json.dump({i:j for i,j in enumerate(list(map(np.float64,model.predict(214))))}, json_file)
