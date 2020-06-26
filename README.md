# GC_GP_food_recommendation
Gachon Univ. graduation project   
food recommendation system with data cleaning model

# Usage   

### install package  
Create enviorment with python 3.6   
Use KoBERT package in https://github.com/SKTBrain/KoBERT.git
    
    pip install -r requirement.txt  
    conda install -c maciejkula -c pytorch spotlight
    git clone https://github.com/SKTBrain/KoBERT.git
    cd KoBERT 
    pip install .   
    cd ..   

***
### data crawling
crawl review data of 60 stores in yogiyo and formatting dataset to our model

    python Clawling.py
    python grouping.py

__Crawling.py__ is crawl data in yogiyo and return data/crawl_data.csv   
__grouping.py__ is change format of crawl data and return test_data, train_data in data floder

***
### data cleaning
Data cleaning with bert or rnn classifier   
#### Train   
Save best epoch's model parameter in cleaning_model   

    python train_cleaning.py --model_mode bert --batch_size 64 --num_epoch 30   
    
>model_mode: setting model mode of data cleaing model[bert/rnn]   
>batch_size: train batch_size   
>num_epoch: train epoch number   
   
#### test   
Testing review data cleaning   

    python test_cleaning.py --model_mode bert --data_tsv data/test_data.txt --data_excel data/test_data.xlsx \    
    --model_file cleaning_model/BERT_27.model --save_file test_result.xlsx 

>model_mode: setting model mode of data cleaing model[bert/rnn]     
>data_tsv: tsv data path, make in __grouping.py__   
>data_excel: tsv data path, make in __grouping.py__   
>model_file: load model prameter path   
>save_file: result file name, save in cleaning_result folder   
***
### recommendation
Recommendation with data cleaning result   

    python recommendation_train.py --model_mode ifm --save_model_file ifm_model.pickle \
    --test_path cleaning_result/cleaning_test_bert.xlsx --train_path data/train_data.xlsx \
    --cleaning_train True --cleaning_test True --n_iter 5
    
>model_mode: setting model mode of recommendation model[ifm/efm/cnn]      
>save_model_file: save file name, save in recommend_model folder   
>test_path: test file path   
>train_path: train file path   
>cleaning_train: set train dataset cleaning   
>cleaning_test: set test dataset cleaning   
>n_iter: model training iteration number   

# RESULT   
### Data cleaning performace   
| model | Accuracy |
|:-----:|:--------:|
| RNN   | 71.32    |
| BERT  | 78.13    |   
***
### Recommendation model performance   
|              model              | Accuracy |
|:-------------------------------:|:--------:|
| Explicit Feedback Factorization |   0.012  |
| Implicit Feedback Factorization |   0.073  |
|             Wavenet             |   0.024  |   
***
### Recommendation System performance with data cleaning   
|           model           | Accuracy |
|:-------------------------:|:--------:|
| Wavenet w/o Data cleaning |   0.020  |
|  Wavenet w/ Data cleaning |   0.060  |
|   IFM w/o Data cleaning   |   0.057  |
|    IFM w/  Data clening   |   0.130  |

