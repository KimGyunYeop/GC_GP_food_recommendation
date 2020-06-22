import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
import gluonnlp as nlp
from kobert.pytorch_kobert import get_pytorch_kobert_model
import mxnet.gluon as gl

class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))


class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size=768,
                 num_classes=2,
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate

        self.classifier = nn.Linear(hidden_size, num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)

        _, pooler = self.bert(input_ids=token_ids, token_type_ids=segment_ids.long(),
                              attention_mask=attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)

class RNNDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, tokenizer, max_len,
                 pad, pair):
        _, vocab = get_pytorch_kobert_model()
        tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)
        self.sentences = np.array(list(gl.data.SimpleDataset([tok.convert_tokens_to_ids(tok(i[sent_idx])) for i in dataset]).transform(nlp.data.PadSequence(max_len))),dtype=np.int32)
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i],self.labels[i])

    def __len__(self):
        return (len(self.labels))


class RNNClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size=768,
                 num_classes=2,
                 dr_rate=None,
                 params=None):
        super(RNNClassifier, self).__init__()
        _, vocab = get_pytorch_kobert_model()
        self.dr_rate = dr_rate
        self.embedding = nn.Embedding(len(vocab.token_to_idx), 100)
        self.rnn = nn.RNN(100, hidden_size, batch_first=True)
        self.classifier = nn.Linear(hidden_size, num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, tokens):
        emb = self.embedding(tokens)
        rnn_result, _ = self.rnn(emb)
        rnn_result = rnn_result[:, -1, :]
        return self.classifier(rnn_result)