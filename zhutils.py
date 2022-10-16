from torchtext.datasets import data
import torch
import spacy
from torchtext.data.metrics import bleu_score
from torchtext.data import Dataset
from torchtext.datasets import TranslationDataset
import sentencepiece as spc
import sys
import os
import io


def translate_sentence(model, sentence, german, english, device, max_length=50):
    # Load german tokenizer
    spacy_ger = spacy.load("de_core_news_md")

    # Create tokens using spacy and everything in lower case (which is what our vocab is)
    if type(sentence) == str:
        tokens = [token.text.lower() for token in spacy_ger(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    # Add <SOS> and <EOS> in beginning and end respectively
    tokens.insert(0, german.init_token)
    tokens.append(german.eos_token)

    # Go through each german token and convert to an index
    text_to_indices = [german.vocab.stoi[token] for token in tokens]

    # Convert to Tensor
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)

    outputs = [english.vocab.stoi["<sos>"]]
    for i in range(max_length):
        trg_tensor = torch.LongTensor(outputs).unsqueeze(1).to(device)

        with torch.no_grad():
            output = model(sentence_tensor, trg_tensor)

        best_guess = output.argmax(2)[-1, :].item()
        outputs.append(best_guess)

        if best_guess == english.vocab.stoi["<eos>"]:
            break

    translated_sentence = [english.vocab.itos[idx] for idx in outputs]
    # remove start token
    return translated_sentence[1:]


def bleu(data, model, german, english, device):
    targets = []
    outputs = []

    for example in data:
        src = vars(example)["src"]
        trg = vars(example)["trg"]

        prediction = translate_sentence(model, src, german, english, device)
        prediction = prediction[:-1]  # remove <eos> token

        targets.append([trg])
        outputs.append(prediction)

    return bleu_score(outputs, targets)


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

def chinese_tokenizer_load():
    spc_zh = spc.SentencePieceProcessor()
    spc_zh.Load('{}.model'.format('zh'))        #can't we just do spc_zh.Load('zh.model')
    return spc_zh

def english_tokenizer_load():
    spc_en = spc.SentencePieceProcessor()
    spc_en.Load('{}.model'.format('en'))
    return spc_en



# class NCDataset(TranslationDataset):

#     """My own Dataset object using corpus from News Commentary v16"""

#     urls = ['http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/training.tar.gz',
#             'http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/validation.tar.gz',
#             'http://www.quest.dcs.shef.ac.uk/'
#             'wmt17_files_mmt/mmt_task1_test2016.tar.gz']
#     name = 'ncv16'
#     dirname = ''

#     @classmethod
#     def splits(cls, exts, fields, root='.data',
#                train='train', validation='val', test='test', **kwargs):
#         """Create dataset objects for splits of the NCv16 dataset.

#         Arguments:
#             exts: A tuple containing the extension to path for each language.
#             fields: A tuple containing the fields that will be used for data
#                 in each language.
#             root: Root dataset storage directory. Default is 'zhen_data'.
#             train: The prefix of the train data. Default: 'train'.
#             validation: The prefix of the validation data. Default: 'devset'.
#             test: The prefix of the test data. Default: 'testset'.
#             Remaining keyword arguments: Passed to the splits method of
#                 Dataset.
#         """

#         if 'path' not in kwargs:
#             expected_folder = os.path.join(root, cls.name)
#             path = expected_folder if os.path.exists(expected_folder) else None
#         else:
#             path = kwargs['path']
#             del kwargs['path']

#         return super(NCDataset, cls).splits(
#             exts, fields, path, root, train, validation, test, **kwargs)