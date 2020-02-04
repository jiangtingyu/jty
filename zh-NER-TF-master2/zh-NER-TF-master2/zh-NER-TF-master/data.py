import sys, pickle, os, random
import numpy as np

## tags, BIO
tag2label = {"O": 0,
             "B-PER": 1, "I-PER": 2,
             "B-LOC": 3, "I-LOC": 4,
             "B-ORG": 5, "I-ORG": 6
             }


#输入train_data文件的路径，读取训练集的语料，输出train_data
def read_corpus(corpus_path):
    """
    read corpus and return the list of samples
    :param corpus_path:
    :return: data
    """
    data = []
    with open(corpus_path, encoding='utf-8') as fr:
        '''lines的形状为['北\tB-LOC\n','京\tI-LOC\n','的\tO\n','...']总共有2220537个字及对应的tag'''
        lines = fr.readlines()
    sent_, tag_ = [], []
    for line in lines:
        if line != '\n':
            # char 与 label之间有个空格
            # line.strip()的意思是去掉每句话句首句尾的空格
            # .split()的意思是根据空格来把整句话切割成一片片独立的字符串放到数组中，同时删除句子中的换行符号\n
            [char, label] = line.strip().split()
            # 把一个个的字放进sent_
            sent_.append(char)
            # 把字后面的tag放进tag_
            tag_.append(label)
        # 一句话结束了，添加到data
        else:
            data.append((sent_, tag_))
            sent_, tag_ = [], []
    """ data的形状为[(['我',在'北','京'],['O','O','B-LOC','I-LOC'])...第一句话
                             (['我',在'天','安','门'],['O','O','B-LOC','I-LOC','I-LOC'])...第二句话  
                              ( 第三句话 )  ] 总共有50658句话"""
    return data

"""由train_data来构造一个(统计非重复字)字典{'第一个字':[对应的id,该字出现的次数],'第二个字':[对应的id,该字出现的次数], , ,}
去除低频词，生成一个word_id的字典并保存在输入的vocab_path的路径下，
保存的方法是pickle模块自带的dump方法，保存后的文件格式是word2id.pkl文件"""
def vocab_build(vocab_path, corpus_path, min_count):
    """

    :param vocab_path:
    :param corpus_path:
    :param min_count:
    :return:
    """
    data = read_corpus(corpus_path)
    word2id = {}
    for sent_, tag_ in data:
        for word in sent_:
            if word.isdigit():  # 检查是否为0-9
                word = '<NUM>'
            elif ('\u0041' <= word <='\u005a') or ('\u0061' <= word <='\u007a'):  #检查是否为英文字母，ASCII码判别
                word = '<ENG>'
            if word not in word2id:
                word2id[word] = [len(word2id)+1, 1]
            else:
                word2id[word][1] += 1
    # 其实前面统计词频的目的就是这里删除低频词，删除完之后也就不用统计词频了
    # 用来统计低频词
    low_freq_words = []
    for word, [word_id, word_freq] in word2id.items():
        if word_freq < min_count and word != '<NUM>' and word != '<ENG>':
            low_freq_words.append(word)
    for word in low_freq_words:
        del word2id[word]

    # 删除低频词后为每个字重新建立id，而不再统计词频
    new_id = 1
    for word in word2id.keys():
        word2id[word] = new_id
        new_id += 1
    word2id['<UNK>'] = new_id
    word2id['<PAD>'] = 0

    print(len(word2id))
    with open(vocab_path, 'wb') as fw:
        # 序列化到名字为word2id.pkl文件
        pickle.dump(word2id, fw)

#输入一句话，生成一个 sentence_id
'''sentence_id的形状为[1,2,3,4,...]对应的sent为['当','希','望','工',程'...]'''
def sentence2id(sent, word2id):
    """

    :param sent:
    :param word2id:
    :return:
    """
    sentence_id = []
    for word in sent:
        if word.isdigit():
            word = '<NUM>'
        elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):
            word = '<ENG>'
        if word not in word2id:
            word = '<UNK>'
        sentence_id.append(word2id[word])
    return sentence_id

#通过pickle模块自带的load方法(反序列化方法)加载输出word2id
def read_dictionary(vocab_path):
    """

    :param vocab_path:
    :return:
    """
    vocab_path = os.path.join(vocab_path)
    with open(vocab_path, 'rb') as fr:
        word2id = pickle.load(fr)
    print('vocab_size:', len(word2id))
    return word2id


#输入vocab，vocab就是前面得到的word2id，embedding_dim=300
def random_embedding(vocab, embedding_dim):
    """

    :param vocab:
    :param embedding_dim:
    :return:
    """
    # 返回一个len(vocab)*embedding_dim=3905*300的矩阵(每个字投射到300维)作为初始值
    # numpy.random.uniform(low,high,size)功能：从一个均匀分布[low,high)中随机采样，注意定义域是左闭右开，即包含low，不包含high.
    # 参数介绍:
    #     
    #     low: 采样下界，float类型，默认值为0；
    #     high: 采样上界，float类型，默认值为1；
    #     size: 输出样本数目，为int或元组(tuple)
    # 类型，例如，size = (m, n, k), 则输出m * n * k个样本，缺省时输出1个值。
    #
    # 返回值：ndarray类型，其形状和参数size中描述一致。
    embedding_mat = np.random.uniform(-0.25, 0.25, (len(vocab), embedding_dim))
    embedding_mat = np.float32(embedding_mat)
    return embedding_mat

#padding,输入一句话，不够标准的样本用pad_mark来补齐
''' 
输入：seqs的形状为二维矩阵，形状为[[33,12,17,88,50]-第一句话
                                 [52,19,14,48,66,31,89]-第二句话
                                                    ] 
输出：seq_list为seqs经过padding后的序列
      seq_len_list保留了padding之前每条样本的真实长度
      seq_list和seq_len_list用来喂给feed_dict
'''

def pad_sequences(sequences, pad_mark=0):
    """

    :param sequences:
    :param pad_mark:
    :return:
    """
    max_len = max(map(lambda x : len(x), sequences))
    seq_list, seq_len_list = [], []
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)
        seq_list.append(seq_)
        seq_len_list.append(min(len(seq), max_len))
    return seq_list, seq_len_list


def batch_yield(data, batch_size, vocab, tag2label, shuffle=False):
    """

    :param data:
    :param batch_size:
    :param vocab:
    :param tag2label:
    :param shuffle:
    :return:
    """
    if shuffle:
        random.shuffle(data)

    seqs, labels = [], []
    for (sent_, tag_) in data:
        sent_ = sentence2id(sent_, vocab)
        label_ = [tag2label[tag] for tag in tag_]

        if len(seqs) == batch_size:
            yield seqs, labels
            seqs, labels = [], []

        seqs.append(sent_)
        labels.append(label_)

    if len(seqs) != 0:
        yield seqs, labels

