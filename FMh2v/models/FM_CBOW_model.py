import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FMCBOWModel(nn.Module):
    def __init__(self, emb_size, emb_dimension, min_batch, v_dim):
        super(FMCBOWModel, self).__init__()
        self.emb_size = emb_size
        self.emb_dimension = emb_dimension
        self.min_batch = min_batch
        self.u_embeddings = nn.Embedding(self.emb_size, self.emb_dimension, sparse=True)  # 定义输入词的嵌入字典形式
        self.w_embeddings = nn.Embedding(self.emb_size, self.emb_dimension, sparse=True)  # 定义输出词的嵌入字典形式
        self._init_embedding()  # 初始化
        self.v = nn.Parameter(torch.randn(v_dim,emb_dimension))
        # self.lin = nn.Linear(self.min_batch,1)
        # self.v_embeddings = nn.Embedding(self.emb_size, 5)

    def _init_embedding(self):
        int_range = 0.5 / self.emb_dimension
        self.u_embeddings.weight.data.uniform_(-int_range, int_range)
        self.w_embeddings.weight.data.uniform_(-0, 0)

    # 用于测试矩阵检验
    def print_for_test(self, pos_u_emb, pos_w_emb, neg_w_emb, s1, s2, s3, n1, n2, n3):
        print('pos_u_emb size:', pos_u_emb.size())
        print('pos_w_emb size:', pos_w_emb.size())
        print('neg_w_emb size:', neg_w_emb.size())
        print('s1 size:', s1.size())
        print('s2 size:', s2.size())
        print('s3 size:', s3.size())
        print('n1 size:', n1.size())
        print('n2 size:', n2.size())
        print('n3 size:', n3.size())

    # 正向传播，输入batch大小得一组（非一个）正采样id，以及对应负采样id
    # pos_u：上下文矩阵, pos_w：中心词矩阵，neg_w：负采样矩阵
    def forward(self, pos_u, pos_w, neg_w):
        pos_u_emb = []  # 上下文embedding
        for per_Xw in pos_u:
            # 上下文矩阵的第一维不同词值不同，如第一个词上下文为c，第二个词上下文为c+1，需要统一化
            per_u_emb = self.u_embeddings(torch.LongTensor(per_Xw))  # 对上下文每个词转embedding
            inter_part1 = torch.mm(per_u_emb, self.v.t())
            inter_part2 = torch.mm(torch.pow(per_u_emb, 2),torch.pow(self.v, 2).t())
            per_u_emb = per_u_emb + 0.5 * torch.sum(torch.pow(inter_part1, 2) - inter_part2)
            per_u_numpy = per_u_emb.data.numpy()  # 转回numpy，好对其求和
            per_u_numpy = np.sum(per_u_numpy, axis=0)
            # per_u = self.lin(per_u_emb)
            # per_u = per_u.reshape(self.emb_dimension)
            # per_u = per_u_numpy.data.numpy()
            # per_u_list = per_u.tolist()  # 为上下文词向量Xw的值
            pos_u_emb.append(per_u_numpy)  # 放回数组
        pos_u_emb = torch.FloatTensor(pos_u_emb)  # 转为tensor 大小 [ mini_batch_size * emb_dimension ]
        pos_w_emb = self.w_embeddings(torch.LongTensor(pos_w))  # 转换后大小 [ mini_batch_size * emb_dimension ]
        neg_w_emb = self.w_embeddings(torch.LongTensor(neg_w))  # 转换后大小 [ negative_sampling_number * mini_batch_size * emb_dimension ]
        # 计算梯度上升（ 结果 *（-1） 即可变为损失函数 ->可使用torch的梯度下降）
        score_1 = torch.mul(pos_u_emb, pos_w_emb).squeeze()  # Xw.T * θu
        score_2 = torch.sum(score_1, dim=1)  # 点积和
        score_3 = F.logsigmoid(score_2)  # log sigmoid (Xw.T * θu)
        neg_score_1 = torch.bmm(neg_w_emb, pos_u_emb.unsqueeze(2)).squeeze()  # Xw.T * θneg(w)
        neg_score_2 = torch.sum(neg_score_1, dim=1)  # 求和∑neg(w) Xw.T * θneg(w)
        neg_score_3 = F.logsigmoid((-1) * neg_score_2)  # ∑neg(w) [log sigmoid (-Xw.T * θneg(w))]
        # L = log sigmoid (Xw.T * θu) + ∑neg(w) [log sigmoid (-Xw.T * θneg(w))]
        loss = torch.sum(score_3) + torch.sum(neg_score_3)
        # print for test
        # self.print_for_test(pos_u_emb, pos_w_emb, neg_w_emb, score_1, score_2, score_3, neg_score_1, neg_score_2,neg_score_3)
        return -1 * loss

    # 存储embedding
    def save_embedding(self, id2word_dict, file_name):
        embedding = self.u_embeddings.weight.data.numpy()
        file_output = open(file_name, 'w', encoding='utf-8')
        file_output.write('%d %d\n' % (self.emb_size, self.emb_dimension))
        for id, word in id2word_dict.items():
            e = embedding[id]
            e = ' '.join(map(lambda x: str(x), e))
            file_output.write('%s %s\n' % (word, e))
        # file_output.write('%d %d' % (self.emb_size, self.emb_dimension))
        # for id, word in id2word_dict.items():
        #     e = embedding[id]
        #     e = ' '.join(map(lambda x: str(x), e))
        #     file_output.write('%s %s' % (word, e))


def test():
    model = FMCBOWModel(100, 4, 3)
    id2word = dict()
    for i in range(100):
        id2word[i] = str(i)
    pos_u = [[1, 2, 3], [0, 1, 2, 3]]
    pos_w = [0, 1]
    neg_w = [[23, 42], [32, 24]]
    model.forward(pos_u, pos_w, neg_w)
    model.save_embedding(id2word, 'test1.txt')


if __name__ == '__main__':
    test()
