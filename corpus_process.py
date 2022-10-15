import csv

if __name__ == "__main__":
    en_sentences = []
    zh_sentences = []

    #populate lists of sentences
    with open('zhen_data/news-commentary-v16.en-zh.tsv', 'r', encoding='utf-8') as file_in:
        tsv_file = csv.reader(file_in, delimiter='\t')
        for line in tsv_file:
            en_sentences.append(line[0] + '\n')
            zh_sentences.append(line[1] + '\n')

    
    assert len(en_sentences) == len(zh_sentences) and len(en_sentences) == 322275
    print('sentences in corpus: {}'.format(len(en_sentences)))
    print('-----------------------------got corpus-----------------------------')


    #write to English corpus
    with open('zhen_data/train.en', 'w', encoding='utf-8') as file_out_en:
        for sent in en_sentences:
            #sent_tokenize(sent)
            # tokenized_sent = [word_tokenize(t) for t in sent_tokenize(sent)]
            # file_out_en.writelines(tokenized_sent)
            if len(sent) > 1:
                file_out_en.writelines(sent)
        

    #write to Chinese corpus
    with open('zhen_data/train.zh', 'w', encoding='utf-8') as file_out_zh:
        for sent in zh_sentences:
            if len(sent) > 1:
                file_out_zh.writelines(sent)

    print('-----------------------------train data finished-----------------------------')