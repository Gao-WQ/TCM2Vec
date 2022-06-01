from utils.headers import *

random.seed(100)
np.random.seed(100)
torch.manual_seed(100)
torch.cuda.manual_seed(100)
torch.cuda.manual_seed_all(100)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
os.environ['PYTHONHASHSEED'] =str(100)

def gelu(x):
    return x*0.5*(1.0 + torch.erf(x / math.sqrt(2.0)))


def fastica(embeddings):
    np.random.seed(1)
    Critical = 0.00001
    Maxcount = 100
    s0 = embeddings.size()[0]
    s1 = embeddings.size()[1]
    s2 = embeddings.size()[2]
    embeddings = embeddings.data.numpy()

    temp_embeds = embeddings.reshape((s0*s1, s2))   #(32,17,100)--->(32*17,100)
    temp_embeds = temp_embeds.T
    R,C = temp_embeds.shape[0],temp_embeds.shape[1]

    average = np.mean(temp_embeds, axis=1)

    for i in range(R):
        temp_embeds[i,:] = temp_embeds[i,:] - average[i]

    Cx = np.cov(temp_embeds)
    value, eigvector = np.linalg.eig(Cx)
    val = value**(-1/2)*np.eye(R, dtype=float)
    White = np.dot(val, eigvector.T)

    Z = np.dot(White, temp_embeds)

    W = 0.5*np.ones([s2,s2])

    for n in range(R):
        count = 0
        WP=W[:,n].reshape(R,1)
        LastWP = np.zeros(R).reshape(R, 1)
        while LA.norm(WP - LastWP, 1) > Critical:
            # print(count," loop :",LA.norm(WP-LastWP,1))
            count = count + 1
            LastWP = np.copy(WP)  # %上次迭代的值
            gx = np.tanh(LastWP.T.dot(Z))  # 行向量

            for i in range(R):
                tm1 = np.mean(Z[i, :] * gx)
                tm2 = np.mean(1 - gx ** 2) * LastWP[i]  # 收敛快
                # tm2=np.mean(gx)*LastWP[i]     #收敛慢
                WP[i] = tm1 - tm2
                # print(" wp :", WP.T )
            WPP = np.zeros(R)  #
            for j in range(n):
                WPP = WPP + WP.T.dot(W[:, j]) * W[:, j]
            WP.shape = 1, R
            WP = WP - WPP
            WP.shape = R, 1
            WP = WP / (LA.norm(WP))
            if (count == Maxcount):
                # print("reach Maxcount，exit loop", LA.norm(WP - LastWP, 1))
                break
        # print("loop count:", count)
        W[:, n] = WP.reshape(R, )
    SZ = W.T.dot(Z)
    SZ = SZ.T
    SZ = torch.from_numpy(SZ)
    SZ = SZ.reshape((s0,s1,s2))
    SZ = SZ.to(torch.float32)
    return SZ



def Y_Loss_1(preds, loss_mode):
    '''
    :param preds:  0,2,4..为正样本，1，3，5。。为副样本
    :param loss_mode:
    :return:
    '''
    y_loss = torch.tensor(0.0)
    # 欧式距离
    if loss_mode == 'Euolidean Distance':
        loss_y_f = nn.PairwiseDistance(p=2)
        for p in range(0, len(preds)-1, 2):
                t1 = preds[p].unsqueeze(0)  # 增添一个维度
                t2 = preds[p + 1].unsqueeze(0)
                y_loss += loss_y_f(t1, t2)[0]  # 欧式距离
        # print(y_loss)

    # KL散度(相对熵)
    elif loss_mode == 'KLDivLoss':
        loss_y_f = nn.KLDivLoss(reduction='batchmean')
        for p in range(0, len(preds)-1, 2):      # 2是副样本个数加一
            y_loss += loss_y_f(preds[p].log(), preds[p+1])
        # print(y_loss)
            # for q in range(p + 1, len(preds)):
            #     # y_loss += torch.mean(torch.abs((torch.log(torch.clip(preds[p]/preds[q], 1e-10, 1.0)))))
            #     y_loss += loss_y_f(preds[p].log(), preds[q])
            #     # TODO:之前的实验室  torch.abs(loss_y_f(preds[p],preds[q]))   (KL1)

    # 平方差
    elif loss_mode == 'MSELoss':
        loss_y_f = nn.MSELoss()
        for p in range(0, len(preds)-1, 2):
            t1 = preds[p].unsqueeze(0)  # 增添一个维度
            t2 = preds[p+1].unsqueeze(0)
            # y_loss += loss_y_f(preds[p], preds[p+1])
            y_loss += loss_y_f(preds[p], preds[p+1])

    # 平滑L1Loss
    elif loss_mode == 'SmoothL1Loss':
        loss_y_f = nn.SmoothL1Loss()
        for p in range(0, len(preds)-1, 2):
            t1 = preds[p].unsqueeze(0)  # 增添一个维度
            t2 = preds[p+1].unsqueeze(0)
            y_loss += loss_y_f(t1, t2)
    # 余弦相似度
    elif loss_mode == 'CosineLoss':
        loss_y_f = nn.CosineSimilarity()
        for p in range(0, len(preds)-1, 2):
            t1 = preds[p].unsqueeze(0)  # 增添一个维度
            t2 = preds[p+1].unsqueeze(0)
            temp = loss_y_f(t1, t2)
            y_loss += (1 - temp[0])  # cos值越大越相似        # TODO：计算出现负数---上面加了绝对值

    return y_loss
