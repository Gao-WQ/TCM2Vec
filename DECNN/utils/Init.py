# !/D:/Anaconda3/
# -*- coding:utf-8 -*-
# author: WWQ time:2020/11/10.

from utils.data_processor import *
from utils.headers import *
from tkinter import _flatten

random.seed(100)
np.random.seed(100)
torch.manual_seed(100)
torch.cuda.manual_seed(100)
torch.cuda.manual_seed_all(100)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
os.environ['PYTHONHASHSEED'] =str(100)


def Init_process(path, params):
    # np.random.seed(100)
    max_len, data = Load_data(path)
    texts = list(data['texts'])
    # texts = TextClean(texts)
    params['max_len'] = max_len

    word2ind = np.load('./data/word2ind_P.npy', allow_pickle=True).tolist()
    ind2word = dict(zip(word2ind.values(), word2ind.keys()))
    params['word2ind'] = word2ind
    params['ind2word'] = ind2word
    vocab = list(word2ind.keys())
    print('vocab_size', len(vocab))

    params['text_words'] = list(set(_flatten(texts)))

    if params['embedding_pretrained'] is not None:
        embeded = np.zeros([len(vocab), params['embedding_dim']])  #
        embed_model = gensim.models.KeyedVectors.load_word2vec_format(params['embedding_pretrained'])
        index = 0
        for i in vocab:
            word2ind[i] = index
            if i == '[PAD]' or i == '[CLS]' or i == '[SEP]' or i == '[MASK]' or i == '[UNK]' or i == '[STA]' or i == '[END]':
                if 'bert' not in params['emb_name']:
                    embed_model[index] = np.random.uniform(0, 1, (params['embedding_dim']))
            else:
                embeded[index] = embed_model[i]
            index += 1
        params['embedding_pretrained'] = embeded

    elif params['emb_name'] == 'One-hot':
        embeded = np.zeros([len(vocab), len(vocab)])
        index = 0
        for i in vocab:
            word2ind[i] = index
            embeded[index][index] = 1
            index += 1
        params['embedding_pretrained'] = embeded

    if params['double_encoder'] == 'D':
        embeded_2 = np.zeros([len(vocab), params['nature_dim']])
        embed_model_2 = gensim.models.KeyedVectors.load_word2vec_format(params['nature_emb'])
        for i in word2ind:
            embeded_2[word2ind[i]] = embed_model_2[i]
        params['nature_emb'] = embeded_2
    params['ind2word'] = ind2word
    params['word2ind'] = word2ind
    labels = list(data['labels'])
    params['vocab_size'] = len(word2ind)
    labels_dict, labels = LabelProcess(labels)  # 处理标签
    x_train, y_train, x_test, y_test = Generate_train_test(texts, labels,
                                                           train_ratio=params['train_ratio'])  # 划分训练集与测试集

    if params['data_augment'] == 'O':
        print('旧训练集样本量：', len(x_train))
        train_data = {'texts': x_train, 'labels': y_train}
        train_data = pd.DataFrame(train_data)
        x_train, y_train = SampleAligned(train_data, params)
        print('新训练集样本量：', len(x_train))
        x_train, y_train, _, _ = Generate_train_test(x_train, y_train, train_ratio=1.0)

    else:
        print('无扩充样本')
        x_train = Text2ind(x_train, word2ind, max_len)  # 文本转变为索引表示
        x_train = torch.tensor(np.array(x_train))

    x_train = torch.tensor(np.array(x_train), dtype=torch.float32)
    y_train = torch.tensor(np.array(y_train), dtype=torch.float32)

    x_test = Text2ind(x_test, word2ind, max_len)
    x_test = torch.tensor(np.array(x_test), dtype=torch.float32)
    y_test = torch.tensor(np.array(y_test), dtype=torch.float32)
    params['num_classes'] = len(labels_dict)

    # if params['shuffle_num'] > 0:
    #     x_train = GenerateViceSamples_1(x_train, shuffle_num=params['shuffle_num'])
    #     y_train = LabelExtend(y_train, shuffle_num=params['shuffle_num'])

    return params, x_train, y_train, x_test, y_test, texts, labels

def Init_process_2(path, params):
    # np.random.seed(100)
    max_len, data = Load_data_2(path)
    texts = list(data['texts'])
    # texts = TextClean(texts)

    params['max_len'] = max_len
    word2ind = np.load('./data/word2ind.npy', allow_pickle=True).tolist()
    ind2word = dict(zip(word2ind.values(), word2ind.keys()))
    params['word2ind'] = word2ind
    params['ind2word'] = ind2word
    vocab = list(word2ind.keys())
    print('vocab_size', len(vocab))
    params['text_words'] = list(set(_flatten(texts)))

    if params['embedding_pretrained'] is not None:
        embeded = np.zeros([len(vocab), params['embedding_dim']])   #
        embed_model = gensim.models.KeyedVectors.load_word2vec_format(params['embedding_pretrained'])
        index = 0
        for i in vocab:
            word2ind[i] = index
            if i == '[PAD]' or i == '[CLS]' or i == '[SEP]' or i == '[MASK]' or i == '[UNK]' or i == '[STA]' or i == '[END]':
                if 'bert' not in params['emb_name']:
                    embed_model[index] = np.random.uniform(0, 1, (params['embedding_dim']))
            else:
                embeded[index] = embed_model[i]
            index += 1
        params['embedding_pretrained'] = embeded

    elif params['emb_name'] == 'One-hot':
        embeded = np.zeros([len(vocab), len(vocab)])
        index = 0
        for i in vocab:
            word2ind[i] = index
            embeded[index][index] = 1
            index += 1

        params['embedding_pretrained'] = embeded

    if params['double_encoder'] == 'D':
        embeded_2 = np.zeros([len(vocab), params['nature_dim']])
        embed_model_2 = gensim.models.KeyedVectors.load_word2vec_format(params['nature_emb'])
        for i in word2ind:
            embeded_2[word2ind[i]] = embed_model_2[i]
        params['nature_emb'] = embeded_2
    params['ind2word'] = ind2word
    params['word2ind'] = word2ind


    labels = list(data['labels'])
    params['vocab_size'] = len(word2ind)
    labels_dict, labels = multi_labels(labels)                # 处理标签
    params['label_dict'] = labels_dict
    print('标签个数：', len(labels_dict))
    x_train,y_train,x_test,y_test = Generate_train_test(texts, labels, train_ratio=params['train_ratio'])               # 划分训练集与测试集

    if params['data_augment'] == 'O':
        print('旧训练集样本量：', len(x_train))
        train_data = {'texts':x_train,'labels':y_train}
        train_data = pd.DataFrame(train_data)
        x_train, y_train = SampleAligned(train_data, params)
        print('新训练集样本量：', len(x_train))
        x_train,y_train,_,_ = Generate_train_test(x_train, y_train, train_ratio=1.0)

    else:
       print('无扩充样本')
       x_train = Text2ind(x_train, word2ind, max_len)       # 文本转变为索引表示
       x_train = torch.tensor(np.array(x_train))

    x_train = torch.tensor(np.array(x_train), dtype=torch.float32)
    y_train = torch.tensor(np.array(y_train), dtype=torch.float32)
    
    x_test = Text2ind(x_test,word2ind, max_len)
    x_test = torch.tensor(np.array(x_test), dtype=torch.float32)
    y_test = torch.tensor(np.array(y_test), dtype=torch.float32)
    params['num_classes'] = len(labels_dict)



    return params, x_train,y_train,x_test,y_test,texts,labels


