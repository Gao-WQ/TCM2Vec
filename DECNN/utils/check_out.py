from utils.headers import *

random.seed(100)
np.random.seed(100)
torch.manual_seed(100)
torch.cuda.manual_seed_all(100)
torch.backends.cudnn.deterministic = True

def Check(y_pred,y_true,texts,params):
    texts = texts.reshape((-1,params['max_len']))
    with open(params['save_path']+'/check_out.txt', 'w', encoding='utf-8') as f:
        for i in range(len(texts)):
            for j in texts[i]:
                if j < len(params['ind2word'])-2:
                    f.write(params['ind2word'][j])
                    f.write(' ')
            f.write('\n')
            f.write('真实标签：')
            f.write(params['class_list'][y_true[i]])
            f.write('\n')
            f.write('预测标签：')
            f.write(params['class_list'][y_pred[i]])
            f.write('\n')
            f.write('\n')

