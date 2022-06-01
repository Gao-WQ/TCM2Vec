from utils.data_processor import *
from tkinter import _flatten
from models.BERT import *

for dim in [100]:
    file_path = './data/unlabeled_data.xlsx'
    fn = '_P'
    word2ind = np.load('./data/word2ind' + fn + '.npy', allow_pickle=True).tolist()

    params = {}

    params['max_pred'] = 3
    params['n_layers'] = 3
    params['n_heads'] = 5
    params['embedding_dim'] = dim
    params['d_ff'] = 100*4
    params['d_k'] = params['d_v'] = 20


    params['epochs'] = 50
    params['ep'] = 20
    params['batch_size'] = 32
    params['learning_rate'] = 0.001
    params['step_size'] = 60
    params['gamma'] = 0.5



    params['max_len'], data = Load_data_unlabel(file_path)
    sentences = data['texts']
    params['sentences'] = TextClean(sentences)

    vocab = list(set(_flatten(list(sentences))))



    ind2word = {i: w for i, w in enumerate(word2ind)}
    params['vocab_size'] = len(word2ind)         # 40

    token_list = list()
    for sentence in sentences:
        arr = [word2ind[s] for s in sentence]
        token_list.append(arr)
    params['word2ind'] = word2ind
    params['ind2word'] = ind2word
    params['token_list'] = token_list

    params['max_len'] = params['max_len'] *2 + 1

    batch = make_data(params)
    input_ids, segment_ids, masked_tokens, masked_pos, isNext = zip(*batch)

    dataset = MyDataSet(input_ids, masked_tokens, masked_pos)
    dataloader = Data.DataLoader(dataset, batch_size=1, shuffle=True)

    model = BERT(params)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adadelta(model.parameters(), lr=0.001)

    if __name__ == '__main__':
        for epoch in range(params['epochs']):
            ep = 1
            for input_ids, masked_tokens, masked_pos in dataloader:

                logits_lm = model(input_ids, masked_pos)

                loss_lm = criterion(logits_lm.view(-1, params['vocab_size']), masked_tokens.view(-1))  # for masked LM
                loss_lm = (loss_lm.float()).mean()

                loss = loss_lm

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                ep = ep + 1
            ep = 1

        embeds = model.extract(torch.LongTensor([i for i in range(len(ind2word))]))
        embeds = embeds.data.numpy()
        with open('./data/embed/bert_embed'+'_'+ str(params['embedding_dim'])+ fn +'.txt', 'w', encoding='utf-8') as f:
            s = str(len(params['ind2word'])) + " " + str(params['embedding_dim']) + "\n"
            f.write(s)
            for i in params['word2ind']:
                s = i
                for j in range(params['embedding_dim']):
                    s = s + ' ' + str(embeds[params['word2ind'][i]][j])
                f.write(s)
                f.write("\n")