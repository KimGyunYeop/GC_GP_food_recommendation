import torch
import gluonnlp as nlp
from tqdm import tqdm, tqdm_notebook
import pandas as pd
import os

from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model


from model import BERTDataset,BERTClassifier,RNNDataset,RNNClassifier
from arg_parser import cleaning_test

args = cleaning_test()
device = torch.device("cuda:1")

## Setting parameters
model_mode = args.model_mode
save_path = os.path.join("cleaning_result",args.save_file)
model_path = args.model_path
max_len = args.max_len
batch_size = args.batch_size
warmup_ratio = args.warmup_ratio
num_epochs = args.num_epochs
max_grad_norm = args.max_grad_norm
log_interval = args.log_interval
learning_rate = args.learning_rate
data_tsv = args.data_tsv
data_excel = args.data_excel

bertmodel, vocab = get_pytorch_kobert_model()

df = pd.read_excel(data_excel_path)
dataset_test = nlp.data.TSVDataset(data_tsv_path, field_indices=[1,2], num_discard_samples=1)

tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

if model_mode.lower() == "bert":
    data_test = BERTDataset(dataset_test, 0, 1, tok, max_len, True, False)
    model = BERTClassifier(bertmodel,  dr_rate=0.5).to(device)
elif model_mode.lower() == "rnn":
    data_test = RNNDataset(dataset_test, 0, 1, tokenizer, max_len, True, False)
    model = RNNClassifier(bertmodel,  dr_rate=0.5).to(device)

test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, num_workers=5)

def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc

test_acc = 0.0
result = []
model.load_state_dict(torch.load(model_path))


if model_mode.lower() == "bert":
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(test_dataloader)):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)

        max_vals, max_indices = torch.max(out, 1)
        test_acc += calc_accuracy(out, label)
        result = result + max_indices.tolist()

elif model_mode.lower() == "rnn":
    for batch_id, (tokens, label) in enumerate(tqdm(test_dataloader)):
        tokens = tokens.long().to(device)
        label = label.long().to(device)
        out = model(tokens)

        max_vals, max_indices = torch.max(out, 1)
        test_acc += calc_accuracy(out, label)
        result = result + max_indices.tolist()

df["fault"] = result
df.to_excel(save_path)