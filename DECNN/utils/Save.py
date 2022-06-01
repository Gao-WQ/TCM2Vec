from utils.headers import *
from tensorboard.backend.event_processing import event_accumulator
from tqdm import tqdm

def saveFile(params):
    save_path = params['save_path']
    dir = os.listdir(save_path + '/log')
    for i in dir:
        if 'events' in i:
            name = i
    events = save_path + '/log/' + name
    print(events)
    event_data = event_accumulator.EventAccumulator(events)
    event_data.Reload()
    print(event_data.scalars.Keys())
    keys = event_data.scalars.Keys()[:]
    df = pd.DataFrame(columns=keys)
    for key in tqdm(keys):
        if 'acc' in key:
            df[key] = pd.DataFrame(event_data.Scalars(key)).value
        if 'loss' in key:
            df[key] = pd.DataFrame(event_data.Scalars(key)).value
        if 'precision' in key:
            df[key] = pd.DataFrame(event_data.Scalars(key)).value
        if 'recall' in key:
            df[key] = pd.DataFrame(event_data.Scalars(key)).value
        if 'f1' in key:
            df[key] = pd.DataFrame(event_data.Scalars(key)).value

    len_df = len(df)
    for key in keys:
        df.loc[len_df, key] = df.loc[:len_df, key].mean()
    for key in keys:
        df.loc[len_df + 1, key] = df.loc[:len_df + 1, key].max()
    save_excel = save_path + '/acc.xlsx'
    print(save_excel)
    df.to_excel(save_excel, index=False)

    with open(save_path+'/params.txt', 'w', encoding='utf-8') as f:
        for i in params.keys():
            f.write(i + '：' + str(params[i]) + '\n')


def saveFile_2(params):
    save_path = params['save_path']
    dir = os.listdir(save_path + '/log')
    for i in dir:
        if 'events' in i:
            name = i
    events = save_path + '/log/' + name
    print(events)
    event_data = event_accumulator.EventAccumulator(events)
    event_data.Reload()
    print(event_data.scalars.Keys())
    keys = event_data.scalars.Keys()[:]
    df = pd.DataFrame(columns=keys)
    for key in tqdm(keys):

        if 'loss' in key:
            df[key] = pd.DataFrame(event_data.Scalars(key)).value
        if 'prec' in key:
            df[key] = pd.DataFrame(event_data.Scalars(key)).value
        if 'rec' in key:
            df[key] = pd.DataFrame(event_data.Scalars(key)).value
        if 'f1' in key:
            df[key] = pd.DataFrame(event_data.Scalars(key)).value
    len_df = len(df)
    for key in keys:
        df.loc[len_df, key] = df.loc[:len_df, key].mean()
    for key in keys:
        df.loc[len_df + 1, key] = df.loc[:len_df + 1, key].max()
    save_excel = save_path + '/acc.xlsx'
    print(save_excel)
    df.to_excel(save_excel, index=False)

    with open(save_path+'/params.txt', 'w', encoding='utf-8') as f:
        for i in params.keys():
            f.write(i + '：' + str(params[i]) + '\n')

def SaveEmbededPic(model, params, words):
    model.eval()
    embeds = model.extract(torch.LongTensor([i for i in range(len(params['ind2word']))]))
    embeds = embeds.data.numpy()
    embeds_reduced = PCA(n_components=2).fit_transform(embeds)
    embed_dim = len(embeds[1])
    labels = list(params['ind2word'].values())
    with open(params['save_path'] + '/embed.txt', 'w', encoding='utf-8') as f:
        s = str(len(params['ind2word'])) + " " + str(embed_dim) + "\n"
        f.write(s)
        for i in params['word2ind']:
            s = i
            for j in range(embed_dim):
                s = s + ' ' + str(embeds[params['word2ind'][i]][j])
            f.write(s)
            f.write("\n")

    plt.figure()
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    flg = plt.figure(figsize=(8, 5),dpi=600)
    ax = flg.gca()
    ax.set_facecolor('black')
    ax.plot(embeds_reduced[:, 0], embeds_reduced[:, 1], '.', markersize=1, alpha=1, color='white')   # 所有的点
    ax.set_xlim([-0.5, 0.5])
    ax.set_ylim(([-0.5, 0.5]))

    if params['file'] == 'StackOverflow':
        words = ['SVN','jSON','HTTPS','Oracle','Magento','Hibernate']
    elif params['file'] == 'Biomedical':
        words = ['biological','escherichia','insulin','pulmonary','lymphocytes']
    elif params['file'] == 'THUCNEWS':
        words = ['基金','开盘','证券','硕士','游戏','台湾']
    elif params['file'] == 'TCM':
        words = ['红花','诃子','茯苓','桑叶','蟾酥','半夏']

    elif params['file'] == 'multi_TCM':
        words = ['红花', '诃子', '茯苓', '桑叶', '僵蚕', '半夏']


    # 计算相似性
    write_words = [params['word2ind'][w] for w in words]
    all_words = {}
    for i in write_words:
        similarity = [cosine_similarity(np.array(embeds[i]),j) for j in embeds]
        similarity = np.array(similarity)
        sort_simi = np.argsort(similarity)  # 下标
        most_simi = sort_simi[-5:]
        all_words[i] = most_simi
        # all_words[i] = sort_simi
    num = 0
    colors = ['red', 'yellow', 'orange', 'purple', 'cyan', 'blue']
    # zhfont1 = matplotlib.font_manager.FontProperties(fname='H:\\Anaconda\\Lib\\site-packages\\matplotlib\\mpl-data\\fonts\\ttf\\simhei.ttf', size=10)
    for w in all_words.keys():
        w = int(w)
        embed = embeds[w]
        word = params['ind2word'][w]
        plt.plot(embed[0], embed[1], '.', alpha=1, color=colors[num])
        # plt.text(embed[0], embed[1], word, fontproperties=zhfont1, alpha=1, color=colors[num])
        plt.text(embed[0], embed[1], word, alpha=1, color=colors[num])
        for j in all_words[w]:
            j = int(j)
            embed_ = embeds[j]
            word = params['ind2word'][j]
            plt.plot(embed_[0], embed_[1], '.', alpha=1, color=colors[num])
            # plt.text(embed_[0], embed_[1], word, fontproperties=zhfont1, alpha=1, color=colors[num])
        num += 1
    plt.savefig(params['save_path'] + '/embed.jpg')
    plt.close()


def CalcuSimi(params, words):
    model = gensim.models.KeyedVectors.load_word2vec_format(params['save_path'] + '/embed.txt')
    top_list =  []
    for word in words:
        res = model.most_similar(word, topn=10)
        top_list.append(res)

    with open(params['save_path'] + '/embed_simi.txt', 'w', encoding='utf-8') as f:
        for i in range(len(words)):
            f.write(words[i] + '\n')
            for j in top_list[i]:
                f.write(j[0] + ':' + str(j[1]) + '\n')
            f.write('\n')

def cosine_similarity(vector1, vector2):
  dot_product = 0.0
  normA = 0.0
  normB = 0.0
  for a, b in zip(vector1, vector2):
    dot_product += a * b
    normA += a ** 2
    normB += b ** 2
  if normA.all() == 0.0 or normB.all() == 0.0:
    return 0
  else:
    return np.round(dot_product / ((normA**0.5)*(normB**0.5)) * 100, 2)


