import csv
import config

if __name__ == "__main__":
    en_sentences = []
    zh_sentences = []

    #populate lists of sentences
    if config.zh_first == True:
        zh_col = 0
        en_col = 1
    else:
        zh_col = 1
        en_col = 0
    with open('.data/wtv2/wikititles-v2.zh-en.tsv', 'r', encoding='utf-8') as file_in:
        tsv_file = csv.reader(file_in, delimiter='\t')
        for line in tsv_file:
            en_sentences.append(line[en_col] + '\n')
            zh_sentences.append(line[zh_col] + '\n')

    
    #assert len(en_sentences) == len(zh_sentences) and len(en_sentences) == 322275
    print('sentences in corpus: {}'.format(len(en_sentences)))
    print('-----------------------------got corpus-----------------------------')


    # #write to English corpus
    # with open(config.tgt_train_path, 'w', encoding='utf-8') as file_out_en:
    #     for sent in en_sentences:
    #         #sent_tokenize(sent)
    #         # tokenized_sent = [word_tokenize(t) for t in sent_tokenize(sent)]
    #         # file_out_en.writelines(tokenized_sent)
    #         #if len(sent) > 1:
    #         file_out_en.writelines(sent)

    with open(config.tgt_small_train_path, 'w', encoding='utf-8') as file_out_en:
        max_len = 30000
        cur_len = 0
        for sent in en_sentences:
            cur_len += 1
            #print("writing to: " + config.tgt_small_train_path + "\nwith line: " + sent)
            file_out_en.writelines(sent)
            if cur_len >= max_len:
                break
        

    # #write to Chinese corpus
    # with open(config.src_train_path, 'w', encoding='utf-8') as file_out_zh:
    #     for sent in zh_sentences:
    #         #if len(sent) > 1:
    #         file_out_zh.writelines(sent)

    with open(config.src_small_train_path, 'w', encoding='utf-8') as file_out_zh:
        max_len = 30000
        cur_len = 0
        for sent in zh_sentences:
            cur_len += 1
            file_out_zh.writelines(sent)
            if cur_len >= max_len:
                break

    print('-----------------------------train data finished-----------------------------')