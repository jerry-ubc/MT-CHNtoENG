from json import encoder
from unicodedata import bidirectional
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import Multi30k                         #German -> English dataset
from torchtext.data import Field, BucketIterator
import numpy as np                                              #loss plots
import spacy                                                    #tokenizer
import random
from torch.utils.tensorboard import SummaryWriter  #prints to tensorboard
from utils import save_checkpoint, load_checkpoint
#from utils import translate_sentence, bleu, save_checkpoint, load_checkpoint

#spacy_zh = spacy.load('zh_core_web_md')
spacy_ge = spacy.load('de_core_news_md')
spacy_en = spacy.load('en_core_web_md')

def tokenizer_ge(text):
    return [tok.text for tok in spacy_ge.tokenizer(text)]

def tokenizer_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

german = Field(tokenize=tokenizer_ge, lower = True,
               init_token='<sos>', eos_token='<eos>')

english = Field(tokenize=tokenizer_en, lower = True,
               init_token='<sos>', eos_token='<eos>')

train_data, dev_data, test_data = Multi30k.splits(exts=('.de', '.en'),
                                                  fields=(german, english))

german.build_vocab(train_data, max_size=10000, min_freq=2)
english.build_vocab(train_data, max_size=10000, min_freq=2)

class Encoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, num_layers, p):
        #input_size: size of vocabulary (German)
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(input_size, embed_size)               #(input) input_size -> embed_size (output)
        #WITH attention version
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, bidirectional=True, dropout=p)

        #with attention, fully connected layers are 2*hidden_size due to bidirectionality
        self.fc_hidden = nn.Linear(hidden_size*2, hidden_size)
        self.fc_cell = nn.Linear(hidden_size*2, hidden_size)

        #WITHOUT attention version
        #self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, dropout=p)  #(input) embed_size -> input_size (output)

    def forward(self, x):
        #x: vector of indices to represent tokenized words/word components
        #   unpacking the information, you get a batch_size of sentences
        #x: (seq_length, batch_size)

        embedding = self.dropout(self.embedding(x))     #embedding: (seq_len, N, embed_size)
        encoder_states, (hidden, cell) = self.rnn(embedding)   #outputs: (seq_len, N, hidden_size)
        hidden = self.fc_hidden(torch.cat((hidden[0:1], hidden[1:2]), dim=2))  #hidden[0:1] represents forward direction, [1:2] represents backwards
        #hidden: (2, batch_size, hidden_size) -> (batch_size, hidden_size * 2)
        cell = self.fc_cell(torch.cat((cell[0:1], cell[1:2]), dim=2))

        #encoder_states contains information about hidden states for each timestep
        #whereas hidden contains information about rightmost hidden state
        #each timestep is necessary to calculate attention across entire sentence
        return encoder_states, hidden, cell

class Decoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, output_size, num_layers, p):
        #note input_size = output_size
        #input_size = size of English vocabulary
        #output is probabilities of each word in vocabulary
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(input_size, embed_size)
        #WITH attention version
        self.rnn = nn.LSTM(2*hidden_size + embed_size, hidden_size, num_layers, dropout=p)
        self.energy = nn.Linear(hidden_size*3, 1)   
        #hidden_size*3 because add hidden states from encoder + hidden from previous step in decoder
        self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU()

        #WITHOUT attention version
        #self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, dropout=p)
        #hidden size of encoder and decoder are the same
        self.fc = nn.Linear(hidden_size, output_size)
        #fc means fully connected

    def forward(self, x, encoder_states, hidden, cell):
        #x: (batch_size), but we want (1, N) because decoder predicts 1 word at a time
        #given previous hidden and cell state, predict next work
        #recall encoder is (seq_len, N) since we send in entire German sentence as input
        x = x.unsqueeze(0)

        embedding = self.dropout(self.embedding(x))                     #embedding: (1, N, embed_size)
        sequence_length = encoder_states.shape[0]
        h_reshaped = hidden.repeat(sequence_length, 1, 1)
        energy = self.relu(self.energy(torch.cat((h_reshaped, encoder_states), dim=2)))
        #energy: (hidden_size*3, 1) achieved by encoder_states with (hidden_size*2) and h_reshaped with (hidden_size*1)
        attention = self.softmax(energy)                    #attention: (seq_length, N, 1)
        attention = attention.permute(1, 0, 2)              #(seq_len, N, 1) -> (N, seq_length, 1) DOUBLE CHECK THESE DIMENSIONS ARE THEY RIGHT?????????
        encoder_states = encoder_states.permute(1, 0, 2)    #(seq_len, N, hidden_size*2) -> (N, seq_len, hidden_size*2)
        context_vector = torch.bmm(attention, encoder_states)
        context_vector = context_vector.permute(1, 0, 2)



        outputs, (hidden, cell) = self.rnn(embedding, (hidden, cell))   #outputs: (1, N, hidden_size)
        predictions = self.fc(outputs)                                  #predictions: (1, N, vocab_length)
        predictions = predictions.squeeze(0)                            #want (N, vocab_length) because ??????????????????????
        return predictions, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_force_ratio=0.5):
        #since model will sometimes predict incorrectly and use that incorrect prediction
        #as input for next word, fix it so it's not getting completely derailed
        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = len(english.vocab)

        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)
        #we predict 1 word at-a-time for a batch
        #each prediction is a vector with length equal to entire vocabulary size
        #so 1 output will be sized: (batch_size, target_vocab_size)
        #add additional outputs on dim=0

        hidden, cell = self.encoder(source)
        x = target[0]   #x gets start token

        for word in range(1, target_len):
            output, hidden, cell = self.decoder(x, hidden, cell)
            outputs[word] = output

            #output: (N, english_vocab_size), argmax gives highest probability guess
            best_guess = output.argmax(1)
            x = target[word] if random.random() < teacher_force_ratio else best_guess
        return outputs


#training hyperparameters
num_epochs = 20
learning_rate = 0.001
batch_size = 64

#model hyperparameters
load_model = False
device = torch.device('cua' if torch.cuda.is_available() else 'cpu')
input_size_encoder = len(german.vocab)
input_size_decoder = len(english.vocab)
output_size = len(english.vocab)
encoder_embed_size = 300
decoder_embed_size = 300
hidden_size = 1024 
num_layers = 4
enc_dropout = 0.5
dec_dropout = 0.5

#tensorboard
writer = SummaryWriter(f'runs/loss_plot')
step = 0

train_iterator, dev_iterator, test_iterator = BucketIterator.splits((train_data, dev_data, test_data),
                                                                     batch_size = batch_size,
                                                                     sort_within_batch = True,
                                                                     sort_key = lambda x: len(x.src),
                                                                     device = device)   
                                                                     #all sentences vary in length, so BucketIterator prioritizes similar length examples to minimize number of padding
                                                                     #to save computations via sort_key parameter

encoder_net = Encoder(input_size_encoder, encoder_embed_size, hidden_size, num_layers, enc_dropout).to(device)
decoder_net = Decoder(input_size_decoder, decoder_embed_size, hidden_size, output_size, num_layers, dec_dropout).to(device)

model = Seq2Seq(encoder_net, decoder_net).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

pad_idx = english.vocab.stoi['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)   #if receives a pad index, don't compute anything

if load_model:
    load_checkpoint(torch.load('my_checkpoint.pth.ptar'), model, optimizer)

sentence = ''

for epoch in range(num_epochs):
    print(f'Epoch [{epoch} / {num_epochs}]')

    checkpoint = {'state_dict':model.state_dict(), 'optimizer':optimizer.state_dict()}
    save_checkpoint(checkpoint)

    #model.eval()
    #translated_sent = translate_sentence(model, sentence, german, english, device, max_length=50)
    #print(f'Translated: \n {translated_sent}')
    #model.train()

    for batch_idx, batch in enumerate(train_iterator):
        in_data = batch.src.to(device)
        target = batch.trg.to(device)

        output = model(in_data, target) #output: (target_len, batch_size, output_dim)
                                        #currently output represents a target_len of words
        
        #print(type)
        #print(type(output))
        output = output[1:].reshape(-1, output.shape[2])
        #output[1:] removes output[0] as it's the start token
        #reshape turns output from 3D -> 2D, -1 means it's value is inferred

        target = target[1:].reshape(-1) #target: (target_len, batch_size) -> (target_len * batch_size)
        optimizer.zero_grad()
        loss = criterion(output, target)

        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)    #clips gradients to avoid exploding gradients
        optimizer.step()

        writer.add_scalar('Training loss', loss, global_step=step)
        step += 1

#score = bleu(test_data, model, german, english, device)
#print(f'Bleu score {score*100:.2f}')
print("DONE!!!!!!!!!!!!!!")