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
        self.relu = nn.ReLU()
        if self.params['double_encoder'] == 'D':
            self.conv_region = nn.Conv2d(1, params['num_filters'], (3, params['embedding_dim']+params['nature_dim']), stride=1)

        elif self.params['emb_name'] == 'One-hot':
            self.conv_region = nn.Conv2d(1, params['num_filters'], (3, params['vocab_size']), stride=1)

        else:
            self.conv_region = nn.Conv2d(1, params['num_filters'], (3, params['embedding_dim']), stride=1)

        self.conv = nn.Conv2d( params['num_filters'],  params['num_filters'], (3, 1), stride=1)
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))  # top bottom
        self.padding2 = nn.ZeroPad2d((0, 0, 0, 1))  # bottom
        self.fc = nn.Linear(params['num_filters'], params['num_classes'])

    def forward(self, x):
        if self.params['double_encoder'] == 'D':
            x_1 = self.embedding_g(x)
            x_2 = self.embedding_n(x)

            x_2 = nn.RReLU(lower=self.params['r_init'], upper=self.params['r_upper'])(x_2)

            x = torch.cat((x_1, x_2), dim=2)
        else:
            x = self.embedding_g(x)
        x = x.unsqueeze(1)  # [batch_size, 250, seq_len, 1]
        x = self.conv_region(x)  # [batch_size, 250, seq_len-3+1, 1]

        x = self.padding1(x)  # [batch_size, 250, seq_len, 1]
        x = self.relu(x)
        x = self.conv(x)  # [batch_size, 250, seq_len-3+1, 1]
        x = self.padding1(x)  # [batch_size, 250, seq_len, 1]
        x = self.relu(x)
        x = self.conv(x)  # [batch_size, 250, seq_len-3+1, 1]
        while x.size()[2] >= 2:
            x = self._block(x)
        x1 = x.squeeze()  # [batch_size, num_filters(250)]
        x = self.fc(x1)
        return x, x1

    def _block(self, x):
        x = self.padding2(x)
        px = self.max_pool(x)

        x = self.padding1(px)
        x = F.relu(x)
        x = self.conv(x)

        x = self.padding1(x)
        x = F.relu(x)
        x = self.conv(x)

        # Short Cut
        x = x + px
        return x

    def extract(self, inputs):
        if self.params['double_encoder'] == 'D':
            x_1 = self.embedding_g(inputs)
            x_2 = self.embedding_n(inputs)
            embeds = torch.cat((x_1, x_2), dim=1)
        else:
            embeds = self.embedding_g(inputs)
        return embeds