import urllib
import torch
#from base_functions import LSTMTagger
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import numpy as np

class Dataset(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx +=1

    def __len__(self):
        return len(self.word2idx)

class Corpus(object):
    def __init__(self):
        self.dictionary = Dataset()

    def get_data(self, path, batch_size = 20):
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

                    # Tokenize the file content
        ids = torch.LongTensor(tokens)
        token = 0
        with open(path, 'r') as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1
        num_batches = ids.size(0) // batch_size
        ids = ids[:num_batches * batch_size]
        return ids.view(batch_size, -1)



class LSTMTagger(nn.Module):
    def __init__(self,embedding_dim,  hidden_dim,vocab_size ,target_size ):
        super(LSTMTagger, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(input_size = embedding_dim, hidden_size= hidden_dim,num_layers= target_size, batch_first=True)
        # The linear layer that maps from hidden state space to tag space
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, h):
        x = self.word_embeddings(x)
        out, (h,c ) =  self.lstm(x , h)
        out = out.reshape(out.size(0) * out.size(1), out.size(2))
        tag_space = self.linear(out)
        #tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_space, (h,c)

#############################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

embed_size = 128
hidden_size = 1024
num_layers = 1
num_epochs = 5
num_samples = 200     # number of words to be sampled
batch_size = 20
seq_length = 30
learning_rate = 0.002

corpus = Corpus()
ids = corpus.get_data('inputs/train.txt', batch_size)
vocab_size = len(corpus.dictionary)
num_batches = ids.size(1) // seq_length

test_corpus = Corpus()
test_ids = test_corpus.get_data('inputs/test.txt', batch_size)
test_vocab_size =  len(test_corpus.dictionary)
test_num_batches = test_ids.size(1) // seq_length


model = LSTMTagger(embed_size,hidden_size,vocab_size,num_layers).to(device)

loss_function = nn.CrossEntropyLoss()
optimizer =  torch.optim.Adam(model.parameters(), lr=learning_rate)

def detach(states):
    return [state.detach() for state in states]


print(num_batches)
print(test_num_batches)


for epoch in range(num_epochs):
    states = (torch.zeros(num_layers, batch_size, hidden_size).to(device),
              torch.zeros(num_layers, batch_size, hidden_size).to(device))

    for i in range(0, ids.size(1) - seq_length, seq_length):
        # Get mini-batch inputs and targets
        inputs = ids[:, i:i + seq_length].to(device)
        targets = ids[:, (i + 1):(i + 1) + seq_length].to(device)

        # Forward pass
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        states = detach(states)
        outputs, states = model(inputs, states)
        loss = loss_function(outputs, targets.reshape(-1))

        # Backward and optimize
        model.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        step = (i + 1) // seq_length
        if step % 500 == 0:
            print('Epoch [{}/{}], Step[{}/{}], Loss: {:.4f}, Perplexity: {:5.2f}'
                  .format(epoch + 1, num_epochs, step, num_batches, loss.item(), np.exp(loss.item())))


###########################################################(testa)
import math

# Test the model
states = (torch.zeros(num_layers, batch_size, hidden_size).to(device),
          torch.zeros(num_layers, batch_size, hidden_size).to(device))
test_loss = 0.
with torch.no_grad():
    for i in range(0, test_ids.size(1) - seq_length, seq_length):
        # Get mini-batch inputs and targets
        inputs = test_ids[:, i:i + seq_length].to(device)
        targets = test_ids[:, (i + 1):(i + 1) + seq_length].to(device)

        # Forward pass
        states = detach(states)
        outputs, states = model(inputs, states)
        test_loss += loss_function(outputs, targets.reshape(-1)).item()

test_loss = test_loss / test_num_batches
print('-' * 89)
print('test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('-' * 89)

# Generate texts using trained model
with torch.no_grad():
    with open('sample2.txt', 'w') as f:
        # Set intial hidden ane cell states
        state = (torch.zeros(num_layers, 1, hidden_size).to(device),
                 torch.zeros(num_layers, 1, hidden_size).to(device))

        # Select one word id randomly
        prob = torch.ones(vocab_size)
        input = torch.multinomial(prob, num_samples=1).unsqueeze(1).to(device)

        for i in range(num_samples):
            # Forward propagate RNN
            output, state = model(input, state)

            # Sample a word id
            prob = output.exp()
            word_id = torch.multinomial(prob, num_samples=1).item()

            # Fill input with sampled word id for the next time step
            input.fill_(word_id)

            # File write
            word = corpus.dictionary.idx2word[word_id]
            word = '\n' if word == '<eos>' else word + ' '
            f.write(word)

            if (i + 1) % 100 == 0:
                print('Sampled [{}/{}] words and save to {}'.format(i + 1, num_samples, 'sample.txt'))


torch.save(model.state_dict(), 'model_LSTM.ckpt')