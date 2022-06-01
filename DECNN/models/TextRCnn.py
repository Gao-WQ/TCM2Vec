from utils.headers import *
from utils.mechanism import fastica,gelu
torch.manual_seed(100)



class TCM_1(nn.Module):
    def __init__(self, params):
        super(TCM_1, self).__init__()
        self.params = params
        self.embedding_g = nn.Embedding(self.params['vocab_size'],
                                        self.params['embedding_dim'],
                                        padding_idx=0)

        if self.params['emb_name'] == 'One-hot':
            self.embedding_g = nn.Embedding(self.params['vocab_size'],
                                            self.params['vocab_size'],
                                            padding_idx=0,
                                            norm_type=2)
            self.embedding_g.weight.data.copy_(torch.from_numpy(params['embedding_pretrained']))
            self.embedding_g.weight.requires_grad = True

        if self.params['embedding_pretrained'] is not None:
            self.embedding_g.weight.data.copy_(torch.from_numpy(params['embedding_pretrained']))
            self.embedding_g.weight.requires_grad = True

        if self.params['double_encoder'] == 'D':
            self.embedding_n = nn.Embedding(self.params['vocab_size'],
                                            self.params['nature_dim'],
                                            padding_idx=0)
            self.embedding_n.weight.data.copy_(torch.from_numpy(params['nature_emb']))
            self.embedding_n.weight.requires_grad = True

        self.sigmoid = nn.Sigmoid()


        if self.params['double_encoder'] == 'D':
            self.lstm = nn.LSTM(params['embedding_dim'] + params['nature_dim'],
                                params['hidden_num'],
                                num_layers=params['num_layers'],
                                bidirectional=True,
                                batch_first=True,
                                dropout=0.5)
            self.maxpool = nn.MaxPool1d(params['max_len'])
            self.fc = nn.Linear(params['hidden_num'] * 2 + params['embedding_dim'] + params['nature_dim'], params['num_classes'])

        elif self.params['emb_name'] == 'One-hot':
            self.lstm = nn.LSTM(params['vocab_size'],
                                params['hidden_num'],
                                num_layers=params['num_layers'],
                                bidirectional=True,
                                batch_first=True,
                                dropout=0.5)
            self.maxpool = nn.MaxPool1d(params['max_len'])
            self.fc = nn.Linear(params['hidden_num'] * 2 + params['vocab_size'], params['num_classes'])

        else:
            self.lstm = nn.LSTM(params['embedding_dim'],
                                params['hidden_num'],
                                num_layers=params['num_layers'],
                                bidirectional=True,
                                batch_first=True,
                                dropout=0.5)
            self.maxpool = nn.MaxPool1d(params['max_len'])
            self.fc = nn.Linear(params['hidden_num'] * 2 + params['embedding_dim'], params['num_classes'])

    def forward(self, x):
        if self.params['double_encoder'] == 'D':
            x_1 = self.embedding_g(x)
            x_2 = self.embedding_n(x)
            x_2 = nn.RReLU(lower=self.params['r_init'], upper=self.params['r_upper'])(x_2)


            x = torch.cat((x_1, x_2), dim=2)
        else:
            x = self.embedding_g(x)

        out, _ = self.lstm(x)
        out = torch.cat((x, out), 2)
        out = F.relu(out)
        out = out.permute(0,2,1)
        # out = self.maxpool(out).squeeze()
        out = self.maxpool(out)
        out = out.squeeze()
        out_1 = out
        out = self.fc(out)

        return out, out_1
        # return out, out
    def extract(self, inputs):
        if self.params['double_encoder'] == 'D':
            x_1 = self.embedding_g(inputs)
            x_2 = self.embedding_n(inputs)
            embeds = torch.cat((x_1, x_2), dim=1)
        else:
            embeds = self.embedding_g(inputs)
        return embeds