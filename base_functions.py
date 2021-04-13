import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


######################### temp training data ################################
TRAINING_DATA = [
    # Tags are: DET - determiner; NN - noun; V - verb
    # For example, the word "The" is a determiner
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]

word_to_ix = {}
for sent, tags in TRAINING_DATA:
    for word in sent:
        if word not in word_to_ix:  # word has not been assigned an index yet
            word_to_ix[word] = len(word_to_ix)  # Assign each word with a unique index
print(word_to_ix)
tag_to_ix = {"DET": 0, "NN": 1, "V": 2}
################################################################################

EMBEDDING_DIM = 6
HIDDEN_DIM = 6

##############################################################################

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

    def forward(self, x, h):
        embeddings = self.word_embeddings(x)
        out, (h,c ) =  self.lstm(embeddings , h)
        out = out.reshape(out.size(0) * out.size(1), out.size(2))
        tag_space = self.linear(out)
        #tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_space, (h,c)
################################################################# (Train)

# model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
#
# loss_function = nn.NLLLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.1)
#
# with torch.no_grad():
#     input =prepare_sequence(TRAINING_DATA[0][0], word_to_ix)
#     tag_scores = model(input)
#     print(tag_scores)
#
# for epoch in range(300):
#     for sent, tags in TRAINING_DATA:
#         model.zero_grad()
#
#         sent_in = prepare_sequence(sent, word_to_ix)
#         targets = prepare_sequence(tags, tag_to_ix)
#
#         tag_scores = model(sent_in)
#
#         loss = loss_function(tag_scores, targets)
#         loss.backward()
#         optimizer.step()
#
#
# ###################################################################### (validate)
# with torch.no_grad():
#     inputs = prepare_sequence(TRAINING_DATA[0][0], word_to_ix)
#     tag_scores = model(inputs)
#     print(tag_scores)
#
