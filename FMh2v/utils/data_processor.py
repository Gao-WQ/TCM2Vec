from utils.headers import *
from tkinter import _flatten


def Load_text_data(path):
    data = pd.read_excel(path)
    new_texts = []
    max_len = 0
    for i in data['texts']:
        temp = re.findall(r'[\u4e00-\u9fa5]+', i)
        new_texts.append(temp)
        if len(temp) > max_len:
            max_len = len(temp)
    return new_texts, max_len


def Load_data(path):
    '''
    读取文件获取所需数据
    path : 文件路径
    return ： 方剂组成，文本功效
    '''
    data = pd.read_excel(path)
    print('原始样本量:', data.shape[0])
    data.head()
    # 提取所需数据
    col1 = data['texts']
    col2 = data['labels']
    texts = []
    labels = []
    errors = []
    max_len = 0
    for i in range(len(col1)):
        temp = re.findall(r'[\u4e00-\u9fa5]+', col1[i])  # 提取组成
        texts.append(temp)
        if len(temp) > max_len:
            max_len = len(temp)
        temp = col2[i]  # 提取功效
        labels.append(temp)
    data = {'texts':texts,'labels':labels}
    data = pd.DataFrame(data)
    print('最长文本长度：', max_len)
    print('样本量：', len(texts))
    print('标签个数', len(set(labels)))
    return max_len, data

def Load_data_unlabel(path):
    '''
    读取文件获取所需数据
    path : 文件路径
    return ： 方剂组成，文本功效
    '''
    data = pd.read_excel(path)
    print('原始样本量:', data.shape[0])
    data.head()
    # 提取所需数据
    col1 = data['texts']
    texts = []
    max_len = 0
    for i in range(len(col1)):
        # temp = literal_eval(col1[i])  # 提取组成
        temp = re.findall(r'[\u4e00-\u9fa5]+',col1[i])
        texts.append(temp)
        if len(temp) > max_len:
            max_len = len(temp)
    data = {'texts':texts}
    data = pd.DataFrame(data)
    print('最长文本长度：', max_len)
    print('样本量：', len(texts))
    return max_len, data

def Load_data_unlabel_E(path):
    '''
    读取文件获取所需数据
    path : 文件路径
    return ： 方剂组成，文本功效
    '''
    data = pd.read_excel(path)
    print('原始样本量:', data.shape[0])
    data.head()
    # 提取所需数据
    col1 = data['texts']
    texts = []
    max_len = 0
    for i in range(len(col1)):
        temp = re.findall(r'[a-zA-Z0-9]+', col1[i])  # 提取组成
        texts.append(temp)
        if len(temp) > max_len:
            max_len = len(temp)
    data = {'texts':texts}
    data = pd.DataFrame(data)
    print('最长文本长度：', max_len)
    print('样本量：', len(texts))
    return max_len, data

def TextClean(texts):
    herb_dict = pd.read_excel('./data/中药别名库(2).xlsx')
    herb_dict.head()
    names_else = list(herb_dict['别名'])
    names_base = list(herb_dict['中药名'])
    new_texts = []
    for i in texts:
        temp = []
        for j in i:
            nn = j
            if j in names_else:
                index = names_else.index(j)
                nn = names_base[index]
            temp.append(nn)
        new_texts.append(temp)
    return new_texts

def Word2ind(texts):
    vocab = set(_flatten(texts))
    word2ind = {}
    word2ind['<sta>'] = 0
    ind2word = {}
    ind2word[0] = '<sta>'
    c = 1
    for word in vocab:
        if word not in word2ind:
            word2ind[word] = c
            ind2word[c] = word
            c += 1
    word2ind['<end>'] = len(word2ind)
    ind2word[len(word2ind)] = '<end>'
    word2ind['<unk>'] = len(word2ind)
    word2ind['<pad>'] = len(word2ind)
    ind2word[len(ind2word)] = '<unk>'
    ind2word[len(ind2word)] = '<pad>'
    return word2ind,ind2word

def Text2ind(texts, word2ind, max_len):
    indTexts = []
    for text in texts:
        temp = []
        # temp.append(word2ind['<sta>'])
        for word in text:
            if word not in word2ind.keys():
                temp.append(word2ind['<unk>'])
            else:
                temp.append(word2ind[word])
        # temp.append(word2ind['<end>'])
        indTexts.append(temp)

    n_indTexts = []
    for t in indTexts:
        temp = t
        if len(t) < max_len:
            for j in range(max_len - len(t)):
                temp.append(word2ind['<pad>'])
        elif len(t) > max_len:
            big = len(t) - max_len
            # temp = temp[1:-1]
            n_temp = []
            # n_temp.append(word2ind['<sta>'])
            for c in range(big):
                ind = random.randint(0, big-c)
                del temp[ind]
            for n in temp:
                n_temp.append(n)
            # n_temp.append(word2ind['<end>'])
            temp = n_temp
        n_indTexts.append(list(temp))
    n_indTexts = n_indTexts
    return n_indTexts

def GloveUnified(texts, word2ind):
    new_texts = []
    for text in texts:
        temp = [word2ind['<sta>']]
        for word in text:
            temp.append(word2ind[word])
        temp.append(word2ind['<end>'])
        new_texts.append(temp)
    return new_texts

def Unified(texts, min_length, max_length, word2ind):
    '''
    :param texts:   输入
    :param length:  应统一的长度大小
    :return: 统一后的文本
    '''
    new_texts = []
    for text in texts:
        if len(text) < max_length and len(text) > min_length:    # 删去长度过短的处方
            temp = [word2ind['sta']]
            temp.append(word for word in texts)
            temp.append(word2ind['pad'] for i in range(max_length-len(texts)))
            temp.append(word2ind['end'])
            new_texts.append(temp)

        elif len(text) > max_length:
            temp = [word2ind['sta']]
            temp.append(word for word in random.sample(text[1:-1], max_length))
            temp.append(word2ind['end'])
            new_texts.append(temp)
    print('剔除了',len(texts)-len(new_texts),'个过短样本')
    return new_texts

def MergeTexts(texts):
    merge_texts = _flatten(texts)
    return merge_texts


def OneHotEncode(encode):
    len_1 = np.max(encode) + 1
    len_2 = encode.shape[0]
    new_encode = np.zeros(shape=(len_2, len_1))
    c = 0
    for i in range(len(new_encode)):
        new_encode[i][encode[c]] = 1
        c += 1
    return new_encode

def LabelProcess(label):
    labels_dict = {}
    labels = []
    c = 0
    for i in label:

        if i not in labels_dict.keys():
            labels_dict[i] = c
            c += 1
        labels.append(labels_dict[i])
    print('标签个数', len(labels_dict))
    return labels_dict, labels

def Generate_train_test(x, y, train_ratio = 0.9):
    '''
    划分训练集与测试集
    x,y : 总样本
    train_ratio : 训练集、测试集划分
    return ： 训练集，测试集，训练集索引，测试集索引
    '''
    # print('x', len(x))
    # print('y', y.shape)
    np.random.seed(100)
    assert len(x) == len(y),print('error shape!')
    length = len(x)

    # 打乱顺序
    items = [i for i in range(len(x))]
    np.random.shuffle(items)
    # print('item', items[10])

    train_size = int(length * train_ratio)
    train_index = items[:train_size]
    test_index = items[train_size:]

    x_train = [x[i] for i in train_index]
    y_train = [y[i] for i in train_index]
    x_test = [x[i] for i in test_index]
    y_test = [y[i] for i in test_index]

    print('训练集个数：', train_size)
    print('测试集个数：', len(x_test))
    return x_train, y_train, x_test, y_test


def SampleAligned(data,params):
    sample_count = Counter(list(data['labels']))
    max_key = max(sample_count,key=sample_count.get)
    max_value = max(sample_count.values())

    x_1 = data.loc[data['labels'] == max_key]['texts']

    align_text = list(Text2ind(x_1, params['word2ind'], params['max_len']))
    align_y = list(data.loc[data['labels'] == max_key]['labels'].values)

    for i in sample_count:
        if sample_count[i] < max_value:
            temp_data = data.loc[data['labels'] == i]
            temp_x, temp_y = GenerateViceSamples_1(temp_data, params,max_value-sample_count[i])
            for c in range(len(temp_x)):
                align_text.append(temp_x[c])
                align_y.append(temp_y[c])


    return align_text, align_y


def GenerateViceSamples_1(data, params, shuffle_num):
    '''
    每个方剂生成随机乱序副样本
    shuffle_num : 每个文本生成副样本的个数
    data : 文本+标签
    return ： 乱序样本生成后的总样本
    '''
    np.random.seed(100)
    # shuffle_text = np.zeros([text_ind.shape[0]*shuffle_num+text_ind.shape[0], text_ind.shape[1], text_ind.shape[2]])
    texts = list(data['texts'])
    shuffle_text = texts
    labels = list(data['labels'])
    shuffle_label = labels

    # 乱序
    for i in range(shuffle_num):
        random.seed(i)
        c_t = random.randint(0, len(texts)-1)
        temp_t = texts[c_t]
        index = [p for p in range(len(temp_t))]
        random.shuffle(index)
        temp = []
        for p in index:
            temp.append(temp_t[p])
        shuffle_text.append(list(temp))
        shuffle_label.append(labels[c_t])

    texts_ind = Text2ind(shuffle_text, params['word2ind'], params['max_len'])
    return texts_ind, shuffle_label

def LabelExtend(labels, shuffle_num):
    labels = torch.Tensor(labels)
    new_labels = torch.zeros((labels.size(0)*(shuffle_num+1)))
    c = 0
    for label in labels:
        for j in range(shuffle_num+1):
            new_labels[c] = label
            c += 1
    # print('旧样本标签大小', labels.shape)
    # print('新样本标签大小',new_labels.shape)
    return new_labels


# sample IsNext and NotNext to be same in small batch size
def make_data(params):
    batch = []
    positive = negative = 0
    while (positive != params['batch_size'] / 2) or (negative != params['batch_size'] / 2):
        # ==========================BERT 的 input 表示================================
        # 随机取两个句子的index
        tokens_a_index, tokens_b_index = randrange(len(params['sentences'])), randrange(
            len(params['sentences']))  # sample random index in sentences
        # 随机取两个句子
        tokens_a, tokens_b = params['token_list'][tokens_a_index], params['token_list'][tokens_b_index]
        # Token (没有使用word piece): 单词在词典中的编码
        input_ids = [params['word2ind']['[CLS]']] + tokens_a + [params['word2ind']['[SEP]']] + tokens_b + [params['word2ind']['[SEP]']]
        # Segment: 区分两个句子的编码（上句全为0 (CLS~SEP)，下句全为1）
        segment_ids = [0] * (1 + len(tokens_a) + 1) + [1] * (len(tokens_b) + 1)

        # ========================== MASK LM ==========================================
        n_pred = min(params['max_pred'], max(1, int(len(input_ids) * 0.15)))  # 15 % of tokens in one sentence
        # token在 input_ids 中的下标(不包括[CLS], [SEP])
        cand_maked_pos = [i for i, token in enumerate(input_ids)
                          if token != params['word2ind']['[CLS]'] and token != params['word2ind']['[SEP]']]  # candidate masked position
        shuffle(cand_maked_pos)

        masked_tokens, masked_pos = [], []  # 被mask的tokens，被mask的tokens的索引号
        for pos in cand_maked_pos[:n_pred]:  # 随机mask 15% 的tokens
            masked_pos.append(pos)
            masked_tokens.append(input_ids[pos])
            # 选定要mask的词
            if random() < 0.8:  # 80%：被真实mask
                input_ids[pos] = params['word2ind']['[MASK]']
            elif random() > 0.9:  # 10%
                index = randint(0, params['vocab_size'] - 1)  # random index in vocabulary
                while index < 4:  # 不能是 [PAD], [CLS], [SEP], [MASK]
                    index = randint(0, params['vocab_size'] - 1)
                input_ids[pos] = index  # 10%：不做mask，用任意非标记词代替
            # 还有10%：不做mask，什么也不做

        # =========================== Paddings ========================================
        # input_ids全部padding到相同的长度
        n_pad = params['max_len'] - len(input_ids)
        input_ids.extend([params['word2ind']['[PAD]']] * n_pad)
        segment_ids.extend([params['word2ind']['[PAD]']] * n_pad)

        # zero padding (100% - 15%) tokens
        if params['max_pred'] > n_pred:
            n_pad = params['max_pred'] - n_pred
            masked_tokens.extend([0] * n_pad)
            masked_pos.extend([0] * n_pad)

        # =====================batch添加数据, 让正例 和 负例 数量相同=======================
        if tokens_a_index + 1 == tokens_b_index and positive < params['batch_size'] / 2:
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, True])  # IsNext
            positive += 1
        elif tokens_a_index + 1 != tokens_b_index and negative < params['batch_size'] / 2:
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, False])  # NotNext
            negative += 1

    return batch


class MyDataSet(Data.Dataset):
    def __init__(self, input_ids, masked_tokens, masked_pos, ):
        # 全部要转成LongTensor类型
        self.input_ids = torch.LongTensor(input_ids)
        self.masked_tokens = torch.LongTensor(masked_tokens)
        self.masked_pos = torch.LongTensor(masked_pos)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.masked_tokens[idx], self.masked_pos[idx]

