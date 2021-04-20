import torch
from main import LSTMTagger, Dataset, Corpus
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

embed_size = 128
hidden_size = 1024
num_layers = 1
num_epochs = 5
num_samples = 200     # number of words to be sampled
batch_size = 20
seq_length = 30
learning_rate = 0.002

checkpoint = torch.load('model_lstm')
corpus = Corpus()
ids = corpus.get_data('inputs/train.txt', batch_size)
vocab_size = len(corpus.dictionary)

model = LSTMTagger(embed_size,hidden_size,vocab_size,num_layers).to(device)
optimizer =  torch.optim.Adam(model.parameters(), lr=learning_rate)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
model.eval()
x = 1

ids2 = corpus.get_data('inputs/valid.txt', batch_size)

with torch.no_grad():
    with open('sample3.txt', 'w') as f:
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

#
# with torch.no_grad():
#     with open('valid_output.txt', 'w') as f:
#         # Set intial hidden ane cell states
#         state = (torch.zeros(num_layers, 1, hidden_size).to(device),
#                  torch.zeros(num_layers, 1, hidden_size).to(device))
#
#         # Select one word id randomly`
#         prob = torch.ones(vocab_size)
#         input = torch.multinomial(prob, num_samples=1).unsqueeze(1).to(device)
#
#         for i in range(num_samples):
#             # Forward propagate RNN
#             output, state = model(input, state)
#
#             # Sample a word id
#             prob = output.exp()
#             word_id = torch.multinomial(prob, num_samples=1).item()
#
#             # Fill input with sampled word id for the next time step
#             input.fill_(word_id)
#
#             # File write
#             word = corpus.dictionary.idx2word[word_id]
#             word = '\n' if word == '<eos>' else word + ' '
#             f.write(word)
#
#             if (i + 1) % 100 == 0:
#                 print('Sampled [{}/{}] words and save to {}'.format(i + 1, num_samples, 'sample.txt')
#
#
#
#
