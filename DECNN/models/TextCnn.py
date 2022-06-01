from utils.headers import *
# from models import Embedding
from utils.mechanism import fastica,gelu
#
random.seed(100)
np.random.seed(100)
torch.manual_seed(100)
torch.cuda.manual_seed_all(100)
torch.backends.cudnn.deterministic = True

class TCM_1(nn.Module):
    def __init__(self, params):
        super(TCM_1, self).__init__()
        self.params = params
        self.embedding_g = nn.Embedding(self.params['vocab_size'],
                                        self.params['embedding_dim'],
                                        padding_idx=0,
                                        norm_type=2)
        if self.params['emb_name'] == 'One-hot':
            self.embedding_g = nn.Embedding(self.params['vocab_size'],
                                            self.params['vocab_size'],
                                            padding_idx=0,
                                            norm_type=2)
            self.embedding_g.weight.data.copy_(torch.from_numpy(params['embedding_pretrained']))
            self.embedding_g.weight.requires_grad = True

        elif self.params['embedding_pretrained'] is not None:
            self.embedding_g.weight.data.copy_(torch.from_numpy(params['embedding_pretrained']))
            self.embedding_g.weight.requires_grad = True



        if self.params['double_encoder'] == 'D':
            self.embedding_n = nn.Embedding(self.params['vocab_size'],
                                            self.params['nature_dim'],
                                            padding_idx=0)
            self.embedding_n.weight.data.copy_(torch.from_numpy(params['nature_emb']))
            self.embedding_n.weight.requires_grad = True

        self.sigmoid = nn.Sigmoid()
        # self.relu = nn.ReLU()
        # self.prelu = nn.PReLU(init=0.1)
        # self.rrelu = nn.RReLU(lower=0.1, upper=0.5)
        # self.elu = nn.ELU(alpha=0.1)
        # self.softplus = nn.Softplus()

        if self.params['double_encoder'] == 'D':
            self.convs = nn.ModuleList(
                [nn.Conv1d(self.params['embedding_dim']+self.params['nature_dim'], params['num_filters'], k) for k in params['filter_sizes']])

        elif self.params['emb_name'] == 'One-hot':
            self.convs = nn.ModuleList(
                [nn.Conv1d(self.params['vocab_size'], params['num_filters'], k) for k in params['filter_sizes']])

        else:
            self.convs = nn.ModuleList(
                [nn.Conv1d(self.params['embedding_dim'], params['num_filters'], k) for k in params['filter_sizes']])

        self.dropout = nn.Dropout(self.params['dropout'])
        self.fc = nn.Linear(self.params['num_filters'] * len(self.params['filter_sizes']), self.params['num_classes'])


    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x))
        x = F.max_pool1d(x, x.size(2))
        return x

    def forward(self, x):
        if self.params['double_encoder'] == 'D':
            x_1 = self.embedding_g(x)
            x_2 = self.embedding_n(x)

            x_2 = nn.RReLU(lower=self.params['r_init'], upper=self.params['r_upper'])(x_2)

            x = torch.cat((x_1, x_2), dim=2)
        else:

            x = self.embedding_g(x)

        x = x.permute(0,2,1)
        out_1 = torch.cat([self.conv_and_pool(x, conv) for conv in self.convs], 1)
        out_1 = out_1.squeeze(2)
        out = self.dropout(out_1)
        out = self.fc(out)
        return out, out_1

    def extract(self, inputs):
        if self.params['double_encoder'] == 'D':
            x_1 = self.embedding_g(inputs)
            x_2 = self.embedding_n(inputs)
            embeds = torch.cat((x_1, x_2), dim=1)
        else:
            embeds = self.embedding_g(inputs)
        return embeds

class TCM_2(nn.Module):
    def __init__(self, params):
        super(TCM_2, self).__init__()
        self.params = params
        self.embedding = Embedding.Embedding(self.params)
        self.sigmoid = nn.Sigmoid()

        if self.params['double_encoder'] == 'D':
            self.convs = nn.ModuleList(
                [nn.Conv1d(self.params['embedding_dim']+self.params['nature_dim'], params['num_filters'], k) for k in params['filter_sizes']])
        else:
            self.convs = nn.ModuleList(
                [nn.Conv1d(self.params['embedding_dim'], params['num_filters'], k) for k in params['filter_sizes']])

        self.dropout = nn.Dropout(self.params['dropout'])
        self.fc = nn.Linear(self.params['num_filters'] * len(self.params['filter_sizes']), self.params['num_classes'])

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x))
        x = F.max_pool1d(x, x.size(2))
        return x

    def forward(self, x):
        x = self.embedding(x)
        out_1 = torch.cat([self.conv_and_pool(x, conv) for conv in self.convs], 1)
        out_1 = out_1.squeeze(2)
        out = self.dropout(out_1)
        out = self.fc(out)
        return out, out_1