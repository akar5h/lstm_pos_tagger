import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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
#########################################
zz = len(TRAIN_DATA)
TRAIN_DATA = TRAIN_DATA[:int(0.2*zz)]

############################################
tag_to_ix = {unique_tags[i]:i for i in range(len(unique_tags))}
word_to_ix = {}
for sent, tags in TRAIN_DATA:
    for word in sent:
        if word not in word_to_ix:  # word has not been assigned an index yet
            word_to_ix[word] = len(word_to_ix)  # Assign each word with a unique index

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

embed_size = 128
hidden_size = 1024
num_layers = 1
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

model = LSTMTagger(embed_size, hidden_size, len(word_to_ix), len(tag_to_ix)).to(device)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr= learning_rate)


for epoch in range(num_epochs):
    print("Starting Epoch:", epoch)# again, normally you would NOT do 300 epochs, it is toy data
    for sentence, tags in TRAIN_DATA:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Tensors of word indices.
        sentence_in = prepare_sequence(sentence, word_to_ix).to(device)
        targets = prepare_sequence(tags, tag_to_ix).to(device)

        # Step 3. Run our forward pass.
        tag_scores = model(sentence_in)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()


# See what the scores are after training
all_scores = []
with torch.no_grad():
    for sentence, tags in TRAIN_DATA[:10]:
        inputs = prepare_sequence(sentence, word_to_ix)
        tag_scores = model(inputs)
        all_scores.append(tag_scores)
        print("\n", sentence, "\n\tPrediction:", tag_scores , "\n\t Actual tags:", tags)


    # The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
    # for word i. The predicted tag is the maximum scoring tag.
    # Here, we can see the predicted sequence below is 0 1 2 0 1
    # since 0 is index of the maximum value of row 1,
    # 1 is the index of maximum value of row 2, etc.
    # Which is DET NOUN VERB DET NOUN, the correct sequence!


PATH = "model_lstm_v2"
torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, PATH)


