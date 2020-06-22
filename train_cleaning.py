import torch
from torch import nn
import gluonnlp as nlp
from tqdm import tqdm, tqdm_notebook
import os

from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

from transformers import AdamW
from transformers.optimization import WarmupLinearSchedule

from model import BERTDataset,BERTClassifier,RNNDataset,RNNClassifier
from arg_parser import cleaning_train

args = cleaning_train()
device = torch.device("cuda:0")

## Setting parameters
model_mode = args.model_mode
save_path = "cleaning_model"
max_len = args.max_len
batch_size = args.batch_size
warmup_ratio = args.warmup_ratio
num_epochs = args.num_epochs
max_grad_norm = args.max_grad_norm
log_interval = args.log_interval
learning_rate =  args.learning_rate

bertmodel, vocab = get_pytorch_kobert_model()

dataset_train = nlp.data.TSVDataset("data/train_data.txt", field_indices=[1,2], num_discard_samples=1)
dataset_test = nlp.data.TSVDataset("data/test_data.txt", field_indices=[1,2], num_discard_samples=1)

tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

if model_mode.lower() == "bert":
    data_train = BERTDataset(dataset_train, 0, 1, tok, max_len, True, False)
    data_test = BERTDataset(dataset_test, 0, 1, tok, max_len, True, False)
    model = BERTClassifier(bertmodel,  dr_rate=0.5).to(device)
elif model_mode.lower() == "rnn":
    data_train = RNNDataset(dataset_train, 0, 1, tokenizer, max_len, True, False)
    data_test = RNNDataset(dataset_test, 0, 1, tokenizer, max_len, True, False)
    model = RNNClassifier(bertmodel,  dr_rate=0.5).to(device)

train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, num_workers=5)
test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, num_workers=5)

# Prepare optimizer and schedule (linear warmup and decay)
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

t_total = len(train_dataloader) * num_epochs
warmup_step = int(t_total * warmup_ratio)

scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_step, t_total=t_total)

def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc

best_param = 0
best_acc = 0.0
best_epoch = 0

if model_mode.lower() == "bert":
    for e in range(num_epochs):
        train_acc = 0.0
        test_acc = 0.0
        model.train()
        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            valid_length = valid_length
            label = label.long().to(device)
            out = model(token_ids, valid_length, segment_ids)
            loss = loss_fn(out, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            train_acc += calc_accuracy(out, label)
            if batch_id % log_interval == 0:
                print("epoch {} batch id {} loss {} train acc {}".format(e + 1, batch_id + 1, loss.data.cpu().numpy(),
                                                                         train_acc / (batch_id + 1)))
        print("epoch {} train acc {}".format(e + 1, train_acc / (batch_id + 1)))
        model.eval()
        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(test_dataloader)):
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            valid_length = valid_length
            label = label.long().to(device)
            out = model(token_ids, valid_length, segment_ids)
            test_acc += calc_accuracy(out, label)
        print("epoch {} test acc {} | best acc {}".format(e + 1, test_acc / (batch_id + 1), best_acc / (batch_id + 1)))
        if best_acc < test_acc:
            print("new_recode")
            best_acc = test_acc
            best_epoch = e
            best_param = model.state_dict()

    print("best_acc:{}\nbest_epoch:{}".format(best_acc / (batch_id + 1), best_epoch + 1))
    torch.save(best_param, os.path.join(save_path,"BERT_{}.model".format(best_epoch + 1)))
elif model_mode.lower() == "rnn":
    for e in range(num_epochs):
        train_acc = 0.0
        test_acc = 0.0
        model.train()
        for batch_id, (tokens, label) in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            tokens = tokens.long().to(device)
            label = label.long().to(device)
            out = model(tokens)
            loss = loss_fn(out, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            train_acc += calc_accuracy(out, label)
            if batch_id % log_interval == 0:
                print("epoch {} batch id {} loss {} train acc {}".format(e + 1, batch_id + 1, loss.data.cpu().numpy(),
                                                                         train_acc / (batch_id + 1)))
        print("epoch {} train acc {}".format(e + 1, train_acc / (batch_id + 1)))
        model.eval()
        for batch_id, (tokens, label) in enumerate(tqdm(test_dataloader)):
            tokens = tokens.long().to(device)
            label = label.long().to(device)
            out = model(tokens)
            test_acc += calc_accuracy(out, label)
        print("epoch {} test acc {} | best acc {}".format(e + 1, test_acc / (batch_id + 1), best_acc / (batch_id + 1)))
        if best_acc < test_acc:
            print("new_recode")
            best_acc = test_acc
            best_epoch = e
            best_param = model.state_dict()

    print("best_acc:{}\nbest_epoch:{}".format(best_acc / (batch_id + 1), best_epoch + 1))
    torch.save(best_param, os.path.join(save_path,"RNN_{}.model".format(best_epoch + 1)))