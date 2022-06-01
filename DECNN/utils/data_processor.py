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

def Load_data_2(path):
    '''
    读取文件获取所需数据
    path : 文件路径
    return ： 方剂组成，文本功效
    '''
    data = pd.read_excel(path)
    print('原始样本量:', data.shape[0])

    # 提取所需数据
    col1 = data['texts']
    col3 = data['labels']
    texts = []
    labels_1 = []
    labels = []
    errors = []
    max_len = 0
    for i in range(len(col1)):
        temp = re.findall(r'[\u4e00-\u9fa5]+', col1[i])  # 提取组成
        texts.append(temp)
        if len(temp) > max_len:
            max_len = len(temp)
        temp = col3[i]
        labels.append(temp)

    data = {'texts':texts,'labels':labels}
    data = pd.DataFrame(data)
    print('最长文本长度：', max_len)
    print('样本量：', len(texts))
    return max_len, data


def Text2ind(texts, word2ind, max_len):
    indTexts = []
    for text in texts:
        temp = []
        # temp.append(word2ind['<sta>'])
        for word in text:
            if word not in word2ind.keys():
                temp.append(word2ind['[UNK]'])
            else:
                temp.append(word2ind[word])
        # temp.append(word2ind['<end>'])
        indTexts.append(temp)

    n_indTexts = []
    for t in indTexts:
        temp = t
        if len(t) < max_len:
            for j in range(max_len - len(t)):
                temp.append(word2ind['[PAD]'])
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

def multi_labels(labels):
    label_dict = {}
    new = []
    for i in labels:
        temp = re.findall(r'[\u4e00-\u9fa5]+', i)
        for j in temp:
            if j not in label_dict.keys():
                label_dict[j] = len(label_dict)
        new.append(temp)

    label_encode = np.zeros(shape=(len(labels), len(label_dict)))
    for i in range(len(new)):
        for j in new[i]:
            label_encode[i][label_dict[j]] = 1
    return label_dict, label_encode


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