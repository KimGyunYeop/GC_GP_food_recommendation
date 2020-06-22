#-- encoding:utf-8 --
import requests
import pandas as pd
import os

data_path = "data/"
store_data = pd.read_excel(os.path.join(data_path,"가게_LIST_용인.xlsx"))

main_url= "https://www.yogiyo.co.kr/api/v1/reviews/"
store_list = store_data["ID"]
store_name = store_data["storeName"]
review_count = store_data["reviews"]
falut = store_data["event"]
sub_url = "/?count=10&only_photo_review=false&page={}&sort=time"

data = 0
store_index = 0
for store_id,store_name_i,falut in zip(store_list,store_name,falut):
    print(store_id)
    review = []
    for i in range(1,int(review_count[store_index]/20)+1):
        url = main_url+str(store_id)+sub_url.format(str(i))
        html = requests.get(url)
        json_data = html.json()
        for j in range(len(json_data)):
            json_data[j]["store_name"] = store_name_i
            json_data[j]["fault"] = falut

        review.extend(json_data)
    if store_index == 0:
        data = pd.DataFrame(review)
    else:
        data = pd.concat([data,pd.DataFrame(review)],ignore_index=True)
    store_index += 1

#pd.read_json(html.json()).to_excel("output.xlsx")
print(review)

print(data.head)
print(len(data))
data.to_csv(os.path.join(data_path,"clawl_data.csv"),encoding="utf-8")