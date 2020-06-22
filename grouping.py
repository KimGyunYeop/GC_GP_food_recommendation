import pandas as pd
from collections import Counter
import os

data_path = "data"
data = pd.read_csv(os.path.join(data_path,"clawl_data.csv"),encoding="utf-8")

counter = pd.DataFrame(Counter(data["nickname"]).items(),columns=["nickname","count"])
data = data.join(counter.set_index("nickname"), on='nickname')
data = data.loc[data["count"] > 5]
#data = data.loc[data["fault"] == 1]
data = data.dropna(subset=['menu_summary'])
data = data.sort_values(by=['nickname'], axis=0)
data = data.reset_index(drop=True)

a = list()
for i in range(len(data)):
    menu_text = data["menu_summary"][i]
    trans = dict()
    for j in menu_text.split(","):
        if "/" in j:
            name = j.split("/")
            if "(" in name[1]:
                count = int(name[1].split("(")[0])
            else:
                count = int(name[1])
            name = name[0]
            trans[name] = count
    a.append(trans)

data["menu"] = a
data = data.drop(columns=["menu_summary","Unnamed: 0","is_deleted","level","is_mine_review","phone","clean","id","is_menu_visible"])

create_id = dict()
for index,i in enumerate(Counter(data["nickname"]).keys()):
    create_id[i] = index
counter = pd.DataFrame(columns=["nickname", "id"])
counter["nickname"] = create_id.keys()
counter["id"] = create_id.values()
data = data.join(counter.set_index("nickname"), on='nickname')
data = data.drop(columns=["nickname"])

split_data = data.copy()
split_data = split_data.iloc[0:0]
store_menu_dict = dict()
menu_id_dict = dict()
for i in range(len(data)):
    line = data.iloc[i,:]
    if line["store_name"] in store_menu_dict.keys():
        store_menu_dict[line["store_name"]].update(line["menu"].keys())
    else:
        store_menu_dict[line["store_name"]] = set()
        store_menu_dict[line["store_name"]].update(line["menu"].keys())
    for j,k in line["menu"].items():
        a = line.copy()
        a["menu"] = j
        split_data = split_data.append(pd.Series(a))

create_id = dict()
id_menu_dict= dict()
for index,i in enumerate(Counter(split_data["menu"]).keys()):
    create_id[i] = index+1
    id_menu_dict[index+1] = i
counter = pd.DataFrame(columns=["menu", "menu_id"])
counter["menu"] = create_id.keys()
counter["menu_id"] = create_id.values()
split_data = split_data.join(counter.set_index("menu"), on='menu')
split_data = split_data.drop(columns=["menu"])
split_data = split_data.sample(frac=1)

train_data = split_data[:int(len(split_data)*0.7)].reset_index(drop=True)
test_data = split_data[int(len(split_data)*0.7):].reset_index(drop=True)
train_data.to_excel(os.path.join(data_path,"train_data.xlsx"))
test_data.to_excel(os.path.join(data_path,"test_data.xlsx"))

for d,n in zip((train_data,test_data),("train_data.txt","test_data.txt")):
    with open(os.path.join(data_path,n),"w",encoding="utf8") as f:
        f.write("\t".join(["id","document","label"])+"\n")
        for i in range(len(d)):
            line = d.iloc[i,:]
            comment = str(line["comment"])
            comment2 = comment.replace("\n"," ")
            comment2 = comment2.replace("\r"," ")
            str_line = "\t".join([str(i),comment2,str(line["fault"])])
            str_line = str_line+"\n"
            f.write(str_line)