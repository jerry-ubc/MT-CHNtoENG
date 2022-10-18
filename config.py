import torch

#Training hyperparameters
num_epochs = 5
learning_rate = 3e-4
batch_size = 32

#Model hyperparameters
src_vocab_size = 10000
tgt_vocab_size = 10000
embed_size = 512
num_heads = 8
num_encoder_layers = 3       #Attention is all you need uses 6?
num_decoder_layers = 3
dropout = 0.10              #seq2seq usually lower, can play around
max_len = 100               #max sentence length? also used for positional embedding
forward_expansion = 2048    #SHOULD BE 2048? MAYBE? comment says use default value, search it up

src_train_path = '.data/wtv2/train.zh'
tgt_train_path = '.data/wtv2/train.en'
src_small_train_path = '.data/wtv2/train_small.zh'
tgt_small_train_path = '.data/wtv2/train_small.en'
src_dev_path = 'data/wtv2/val.zh'
tgt_dev_path = '.data/wtv2/val.en'
src_test_path = 'data/wtv2/test.zh'
tgt_test_path = '.data/wtv2/test.en'
zh_first = True