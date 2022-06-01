from utils.headers import *


class BertEmbedding(nn.Module):
    def __init__(self,params):
        super(BertEmbedding, self).__init__()
        # embedding_dim:即embedding_dim
        # token embedding
        self.tok_embed = nn.Embedding(params['vocab_size'], params['embedding_dim'])

        # position embedding: 这里简写了,源码中位置编码使用了sin，cos
        #         self.pos_embed = nn.Embedding(maxlen, embedding_dim)
        self.pos_embed = torch.tensor(
            [[pos / (10000.0 ** (i // 2 * 2.0 / params['embedding_dim'])) for i in range(params['embedding_dim'])] for pos in range(params['max_len'])]
        )
        self.pos_embed[:, 0::2] = torch.sin(self.pos_embed[:, 0::2])
        self.pos_embed[:, 1::2] = torch.cos(self.pos_embed[:, 1::2])

        # LayerNorm
        self.norm = nn.LayerNorm(params['embedding_dim'])

    def forward(self, x):  # x 和 pos的shape 都是[batch_size, seq_len]

        #         seq_len = x.size(1)
        #         pos = torch.arange(seq_len, dtype=torch.long)
        # unsqueeze(0): 在索引0处，增加维度--> [1, seq_len]
        # expand: 某个 size=1 的维度上扩展到size
        # expand_as: 把一个tensor变成和函数括号内一样形状的tensor
        #         pos = pos.unsqueeze(0).expand_as(x)     # [seq_len] -> [batch_size, seq_len]

        # 三个embedding相加
        input_embedding = self.tok_embed(x) + nn.Parameter(self.pos_embed, requires_grad=False)

        return self.norm(input_embedding)

    def extract(self,inputs):
        embeds = self.tok_embed(inputs)
        return embeds


# Padding的部分不应该计算概率，所以需要在相应位置设置mask
# mask==0的内容填充1e-9，使得计算softmax时概率接近0
# 在计算attention时，使用
def get_attn_pad_mask(seq_q, seq_k):    # seq_q 和 seq_k 的 shape 都是 [batch_size, seq_len]
    batch_size, seq_len = seq_q.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_q.data.eq(0).unsqueeze(1)              # [batcb_size, 1, seq_len]
    return pad_attn_mask.expand(batch_size, seq_len, seq_len) # [batch_size, seq_len, seq_len]


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention
    """

    def __init__(self, params):
        super(ScaledDotProductAttention, self).__init__()
        self.params = params
    def forward(self, Q, K, V, attn_mask):
        """
        Args:
            Q: [batch_size, n_heads, seq_len, d_k]
            K: [batch_size, n_heads, seq_len, d_k]
            V: [batch_size, n_heads, seq_len, d_k]
        Return:
            self-attention后的张量，以及attention张量
        """
        # [batch_size, n_heads, seq_len, d_k] * [batch_size, n_heads, d_k, seq_len] = [batch_size, n_heads, seq_len, seq_len]
        score = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.params['d_k'])

        # mask==0 is PAD token
        # 我们需要防止解码器中的向左信息流来保持自回归属性。 通过屏蔽softmax的输入中所有不合法连接的值（设置为-∞）
        score = score.masked_fill_(attn_mask, -1e9)  # mask==0的内容填充-1e9，使得计算softmax时概率接近0

        attention = F.softmax(score, dim=-1)  # [bz, n_hs, seq_len, seq_len]
        context = torch.matmul(attention, V)  # [batch_size, n_heads, seq_len, d_k]

        return context


class MultiHeadAttention(nn.Module):
    def __init__(self, params):
        super(MultiHeadAttention, self).__init__()
        self.params = params
        self.W_Q = nn.Linear(params['embedding_dim'], params['d_k'] * params['n_heads'])  # 其实就是[embedding_dim, embedding_dim]
        self.W_K = nn.Linear(params['embedding_dim'], params['d_k'] * params['n_heads'])
        self.W_V = nn.Linear(params['embedding_dim'], params['d_v'] * params['n_heads'])

    def forward(self, Q, K, V,
                attn_mask):  # Q和K: [batch_size, seq_len, embedding_dim], V: [batch_size, seq_len, embedding_dim], attn_mask: [batch_size, seq_len, seq_len]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(Q).view(batch_size, -1, self.params['n_heads'], self.params['d_k']).transpose(1, 2)  # q_s: [batch_size, n_heads, seq_len, d_k]
        k_s = self.W_K(K).view(batch_size, -1, self.params['n_heads'], self.params['d_k']).transpose(1, 2)  # k_s: [batch_size, n_heads, seq_len, d_k]
        v_s = self.W_V(V).view(batch_size, -1, self.params['n_heads'], self.params['d_v']).transpose(1, 2)  # v_s: [batch_size, n_heads, seq_len, d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.params['n_heads'], 1,
                                                  1)  # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, seq_len, d_v], attn_mask: [batch_size, n_heads, seq_len, seq_len]
        context = ScaledDotProductAttention(self.params)(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1,
                                                            self.params['n_heads'] * self.params['d_v'])  # context: [batch_size, seq_len, n_heads, d_v]

        output = nn.Linear(self.params['n_heads'] * self.params['d_v'], self.params['embedding_dim'])(context)

        return nn.LayerNorm(self.params['embedding_dim'])(output + residual)  # output: [batch_size, seq_len, embedding_dim]

class PoswiseFeedForwardNet(nn.Module):  # 前向传播，线性激活再线性
    def __init__(self, params):
        self.params = params
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(self.params['embedding_dim'], self.params['d_ff'])
        self.fc2 = nn.Linear(self.params['d_ff'], self.params['embedding_dim'])

    def forward(self, x):
        # [batch_size, seq_len, embedding_dim] -> [batch_size, seq_len, d_ff] -> [batch_size, seq_len, embedding_dim]
        return self.fc2(gelu(self.fc1(x)))

class EncoderLayer(nn.Module):  # 多头注意力和前向传播的组合
    def __init__(self,params):
        self.params = params
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(self.params)
        self.pos_ffn = PoswiseFeedForwardNet(self.params)

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
                                         enc_self_attn_mask)  # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size, seq_len, embedding_dim]
        return enc_outputs

class BERT(nn.Module):
    def __init__(self, params):
        super(BERT, self).__init__()
        self.params = params
        self.embedding = BertEmbedding(self.params)
        self.layers = nn.ModuleList([EncoderLayer(self.params) for _ in range(self.params['n_layers'])])
        self.fc = nn.Sequential(
            nn.Linear(self.params['embedding_dim'], self.params['embedding_dim']),
            nn.Dropout(0.5),
            nn.Tanh(),
        )
        self.classifier = nn.Linear(self.params['embedding_dim'], 2)
        self.linear = nn.Linear(self.params['embedding_dim'], self.params['embedding_dim'])
        self.activ2 = gelu
        # fc2 is shared with embedding layer
        embed_weight = self.embedding.tok_embed.weight
        self.fc2 = nn.Linear(self.params['embedding_dim'], self.params['vocab_size'], bias=False)
        self.fc2.weight = embed_weight

    # input_ids和segment_ids的shape[batch_size, seq_len]，masked_pos的shape是[batch_size, max_pred]
    def forward(self, input_ids, masked_pos):
        output = self.embedding(input_ids)  # [bach_size, seq_len, embedding_dim]

        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids)  # [batch_size, seq_len, seq_len]
        for layer in self.layers:  # 这里对layers遍历，相当于源码中多个transformer_blocks
            output = layer(output, enc_self_attn_mask)  # output: [batch_size, seq_len, embedding_dim]

        masked_pos = masked_pos[:, :, None].expand(-1, -1, self.params['embedding_dim'])  # [batch_size, max_pred, embedding_dim]
        h_masked = torch.gather(output, 1, masked_pos)  # masking position [batch_size, max_pred, embedding_dim]
        h_masked = self.activ2(self.linear(h_masked))  # [batch_size, max_pred, embedding_dim]
        logits_lm = self.fc2(h_masked)  # [batch_size, max_pred, vocab_size]

        # logits_lm: [batch_size, max_pred, vocab_size], logits_clsf: [batch_size, 2]
        return logits_lm

    def extract(self, inputs):
        embeds = self.embedding.extract(inputs)
        return embeds