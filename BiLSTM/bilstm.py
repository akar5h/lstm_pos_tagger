import torch
import torch.nn as nn
import torch.optim as optim

from torchtext.legacy import data
from torchtext.legacy import datasets

import spacy
import numpy as np

from time import time
import random

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_data():
    TEXT = data.Field(lower = True)
    UD_TAGS = data.Field(unk_token = None)
    PTB_TAGS = data.Field(unk_token = None)

    fields = (("text", TEXT), ("udtags", UD_TAGS), ("ptbtags", PTB_TAGS))
    train_data, valid_data, test_data = datasets.UDPOS.splits(fields)


    print(f"Number of training examples: {len(train_data)}")
    print(f"Number of validation examples: {len(valid_data)}")
    print(f"Number of testing examples: {len(test_data)}")


    MIN_FREQ = 2

    TEXT.build_vocab(train_data,
                     min_freq = MIN_FREQ,
                     vectors = "glove.6B.100d",
                     unk_init = torch.Tensor.normal_)


    UD_TAGS.build_vocab(train_data)
    PTB_TAGS.build_vocab(train_data)

    print(f"Unique tokens in TEXT vocabulary: {len(TEXT.vocab)}")
    print(f"Unique tokens in UD_TAG vocabulary: {len(UD_TAGS.vocab)}")
    print(f"Unique tokens in PTB_TAG vocabulary: {len(PTB_TAGS.vocab)}")

    return TEXT, PTB_TAGS, train_data, test_data, valid_data


def tag_percentage(tag_counts):
    total_count = sum([count for tag, count in tag_counts])
    tag_counts_percentages = [(tag, count, count / total_count) for tag, count in tag_counts]
    return tag_counts_percentages


class BiLSTM(nn.Module):
    def __init__(self  , input_dim, embedding_dim, hidden_dim, output_dim ,n_layers
                 , bidirectional, dropout, pad_idx ):
        super(BiLSTM, self).__init__()
        self.embeddings = nn.Embedding(input_dim, embedding_dim, padding_idx= pad_idx )
        self.LSTM = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers= n_layers,
                            bidirectional= bidirectional,
                            dropout = dropout if n_layers> 1 else 0)
        self.fc = nn.Linear(hidden_dim *2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embeded = self.dropout(self.embeddings(text))

        output, (h,c) = self.LSTM(embeded)
        predictions = self.fc(self.dropout(output))
        return predictions



class BiLSTMTagger():
    def __init__(self, TEXT, PTB_TAGS ):
        self.INPUT_DIM = len(TEXT.vocab)
        self.EMBEDDING_DIM = 100
        self.HIDDEN_LAYERS = 128
        self.OUTPUT_DIM = len(PTB_TAGS.vocab)
        self.N_LAYERS = 2
        self.BIDIRECTIONAL = True
        self.DROPOUT= 0.25
        self.PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
        self.TAG_PAD_IDX = PTB_TAGS.vocab.stoi[PTB_TAGS.pad_token]
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.TAG_PAD_IDX)

        self.model = BiLSTM(self.INPUT_DIM,
                            self.EMBEDDING_DIM,
                            self.HIDDEN_LAYERS,
                            self.OUTPUT_DIM,
                            self.N_LAYERS,
                            self.BIDIRECTIONAL,
                            self.DROPOUT,
                            self.PAD_IDX)

        self.pretrained_embeddings = TEXT.vocab.vectors

        self.optimizer = optim.Adam(self.model.parameters())

        #print(self.pretrained_embeddings.shape)



    def init_weights(self, m):
        for name, param in m.named_parameters():
            nn.init.normal_(param.data, mean= 0 , std= 0.1)\


    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def restore_embeddings(self):
        self.model.embeddings.weight.data.copy_(self.pretrained_embeddings)
        self.model.embeddings.weight.data[self.PAD_IDX] = torch.zeros(self.EMBEDDING_DIM)
        x = 1

    def categorical_accuracy(self, preds, y):
        """
            Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
            """
        max_preds = preds.argmax(dim=1, keepdim=True)  # get the index of the max probability
        non_pad_elements = (y != self.TAG_PAD_IDX).nonzero()
        correct = max_preds[non_pad_elements].squeeze(1).eq(y[non_pad_elements])
        return correct.sum() / torch.FloatTensor([y[non_pad_elements].shape[0]]).to(device)

    def pre_train(self):
        self.model.apply(self.init_weights)
        print("Total Number of Parameters: ",self.count_parameters())
        self.model = self.model.to(device)
        self.criterion = self.criterion.to(device)
        self.pretrained_embeddings = self.pretrained_embeddings.to(device)

        self.restore_embeddings()



    def train(self, iterator):
        self.pre_train()
        epoch_loss = 0
        epoch_accuracy = 0
        self.model.train()

        for batch in iterator:

            text = batch.text
            tags = batch.ptbtags

            self.optimizer.zero_grad()

            predictions = self.model(text)
            predictions = predictions.view(-1, predictions.shape[-1])

            tags = tags.view(-1)

            loss = self.criterion(predictions, tags)

            acc = self.categorical_accuracy(predictions, tags)

            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
            epoch_accuracy += acc.item()

        return epoch_loss / len(iterator), epoch_accuracy / len(iterator)

    def evaluate(self, iterator):
        epoch_loss = 0
        epoch_accuracy = 0
        self.model.eval()

        with torch.no_grad():
            for batch in iterator:
                text = batch.text
                tags = batch.ptbtags

                self.optimizer.zero_grad()

                predictions = self.model(text)
                predictions = predictions.view(-1, predictions.shape[-1])

                tags = tags.view(-1)

                loss = self.criterion(predictions, tags)

                acc = self.categorical_accuracy(predictions, tags)

                epoch_loss += loss.item()
                epoch_accuracy += acc.item()

        return epoch_loss / len(iterator), epoch_accuracy / len(iterator)

    def epoch_time(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs






BATCH_SIZE = 128
TEXT, TAGS, train_data, valid_data, test_data = get_data()
train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
            (train_data, valid_data, test_data),
            batch_size= BATCH_SIZE,
            device=device)

pos_tagger = BiLSTMTagger(TEXT, TAGS)

N_EPOCHS = 10
best_loss = float('inf')
for epoch in range(N_EPOCHS):
    start_time = time()

    train_loss, train_acc = pos_tagger.train(train_iterator)
    valid_loss, valid_acc = pos_tagger.evaluate(valid_iterator)

    end_time = time()
    epoch_mins, epoch_secs = pos_tagger.epoch_time(start_time, end_time)
    if valid_loss < best_loss:
        best_valid_loss = valid_loss
        torch.save(pos_tagger.model.state_dict(), 'model1.pt')

    print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

x = 1

test_loss, test_acc = pos_tagger.evaluate( test_iterator)

print(f'Test Loss: {test_loss:.3f} |  Test Acc: {test_acc*100:.2f}%')


def tag_sentence( tagger,sentence, text_field, tag_field ):
    tagger.model.eval()
    if isinstance(sentence, str):
        nlp = spacy.load('en')
        tokens = [token.text for token in nlp(sentence)]
    else:
        tokens = [token for token in sentence]
    if text_field.lower:
        tokens = [t.lower() for t in tokens]

    numericalized_tokens = [text_field.vocab.stoi[t] for t in tokens]
    unk_idx = text_field.vocab.stoi[text_field.unk_token]
    unks = [t for t, n in zip(tokens, numericalized_tokens) if n == unk_idx]
    token_tensor = torch.LongTensor(numericalized_tokens)
    token_tensor = token_tensor.unsqueeze(-1).to(device)
    predictions = tagger.model(token_tensor)
    top_predictions = predictions.argmax(-1)
    predicted_tags = [tag_field.vocab.itos[t.item()] for t in top_predictions]

    return tokens, predicted_tags, unks



example_index = 1

sentence = vars(train_data.examples[example_index])['text']
actual_tags = vars(train_data.examples[example_index])['ptbtags']

print(sentence)

tokens, pred_tags, unks = tag_sentence(pos_tagger,
                                       sentence,
                                       TEXT,
                                       TAGS)

print("Pred. Tag\tActual Tag\tCorrect?\tToken\n")

for token, pred_tag, actual_tag in zip(tokens, pred_tags, actual_tags):
    correct = '✔' if pred_tag == actual_tag else '✘'
    print(f"{pred_tag}\t\t{actual_tag}\t\t{correct}\t\t{token}")


y= -1