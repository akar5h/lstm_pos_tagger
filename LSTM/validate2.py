import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import pandas as pd


unique_tags =['PRP$', 'WP$', 'JJR', 'VBP', 'FW', 'IN', 'NNS', 'VBD', 'VBG', 'LRB', 'WP', 'RP', 'NNP', 'JJS', 'PDT', 'RBS', 'PRP', 'DT', 'CD', 'NN', 'VBN', 'TO', 'EX', 'POS', 'RB', 'CC', 'RRB', 'WDT', 'UH', '<UNK>', 'NNPS', 'RBR', 'JJ', 'WRB', 'VBZ', 'VB', 'MD']

def get_test_valid():
    df = pd.read_csv("data/kaggle-GMB/ner_dataset.csv", encoding="ISO-8859-1")
    df = df.fillna(method='pad')
    ignore_tags = ['.', '``', '$', ';', '\'', ':', "(", ")", ',',]
    print(df.columns)
    data = []
    grouped = df.groupby(["Sentence #"])
    unique_tags = []
    for number, group in grouped:
        words = group["Word"].to_list()
        tags = group["POS"].to_list()
        tags = [i if i not in ignore_tags else '<UNK>' for i in tags]
        data.append((words, tags))
        unique_tags+=tags
    data_len = len(data)
    train_len = int(data_len * (0.81))
    train_data = data[:train_len]
    test_data = data[train_len:]
    unique_tags = list(set(unique_tags))
    print(unique_tags)

    return  train_data, test_data



def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


TRAIN_DATA, TEST_DATA = get_test_valid()

########################
zz = len(TRAIN_DATA)
TRAIN_DATA = TRAIN_DATA[:int(0.2*zz)]

########################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

embed_size = 128
hidden_size = 1024

num_epochs = 5
num_samples = 200     # number of words to be sampled
batch_size = 20
seq_length = 30
learning_rate = 0.002

class LSTMTagger(nn.Module):
    def __init__(self,embedding_dim,  hidden_dim,vocab_size ,target_size, ):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)


        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(input_size = embedding_dim, hidden_size= hidden_dim,num_layers= target_size, batch_first=True)
        # The linear layer that maps from hidden state space to tag space
        self.linear = nn.Linear(hidden_dim, target_size)

    def forward(self, x):
        embeddings = self.word_embeddings(x)
        lstm_out, _ = self.lstm(embeddings.view(len(x), 1, -1))
        tag_space = self.linear(lstm_out.view(len(x), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

checkpoint = torch.load('model_lstm_v2')

unique_tags =['PRP$', 'WP$', 'JJR', 'VBP', 'FW', 'IN', 'NNS', 'VBD', 'VBG', 'LRB', 'WP', 'RP', 'NNP', 'JJS', 'PDT', 'RBS', 'PRP', 'DT', 'CD', 'NN', 'VBN', 'TO', 'EX', 'POS', 'RB', 'CC', 'RRB', 'WDT', 'UH', '<UNK>', 'NNPS', 'RBR', 'JJ', 'WRB', 'VBZ', 'VB', 'MD']
tag_to_ix = {unique_tags[i]:i for i in range(len(unique_tags))}

word_to_ix2 = {}
for sent, tags in TRAIN_DATA:
    for word in sent:
        if word not in word_to_ix2:  # word has not been assigned an index yet
            word_to_ix2[word] = len(word_to_ix2)  # Assign each word with a unique index

model = LSTMTagger(embed_size, hidden_size, len(word_to_ix2), len(tag_to_ix)).to(device)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr= learning_rate)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
model.eval()

with torch.no_grad():
    for sent, tags in TRAIN_DATA[:10]:
        inputs = prepare_sequence(sent, word_to_ix2).to(device)
        tag_scores = model(inputs)
        print(tag_scores)