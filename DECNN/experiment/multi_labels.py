from utils.data_processor import *
from utils import Init,Labels_Train
from models import TextCnn,TextRCnn,DPCNN,TextRnn


random.seed(100)
np.random.seed(100)
torch.manual_seed(100)
torch.cuda.manual_seed(100)
torch.cuda.manual_seed_all(100)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
os.environ['PYTHONHASHSEED'] =str(100)

for mm in ['TextCnn', 'TextRCnn',  'DPCNN']:
    file_path = './data/multi_labels.xlsx'
    s_p = './save/'+ 'multi_labels/' + mm + '/' +time.strftime('%m-%d_%H.%M', time.localtime())

    params = {}
    params['fn'] = '_P'
    params['file'] = 'multi_TCM'
    params['train_ratio'] = 0.9
    params['hidden_num'] = 100
    params['dropout_lstm'] = True
    params['dropout_out'] = True
    params['activity'] = True
    params['require_improvement'] = 1000    # 如果1000个batch都没有更新 就结束训练
    params['class_list'] = ['活血剂', '固涩剂', '祛湿利水剂', '解表剂', '温里剂', '泻下剂', '熄风剂', '和解剂', '理气剂', '痈疡剂', '消导剂']  # TODO open class_name

    params['label_mode'] = 'labels'
    params['eval_mode'] = 'ONE'
    params['top_k'] = 2

    params['loss_func'] = nn.MultiLabelSoftMarginLoss()  # 3
    params['dropout'] = 0.5
    params['epochs'] = 40
    params['ep'] = 20
    params['batch_size'] = 32
    params['learning_rate'] = 0.001
    params['step_size'] = 60
    params['gamma'] = 0.5

    params['shuffle_num'] = 0
    params['model_name'] = mm   # TODO!!!
    params['embedding_dim'] = 100
    params['nature_dim'] = 23

    # params['embedding_pretrained'] = None
    # params['emb_name'] = 'None'
    # params['emb_name'] = 'word2vec_embed_' + str(params['embedding_dim']) + params['fn']
    # params['emb_name'] = 'bert_embed_' + str(params['embedding_dim']) + params['fn']
    params['emb_name'] = 'FM_embed_50_100'
    params['embedding_pretrained'] = './data/embed/' + params['emb_name'] + '.txt'
    params['nature_emb'] = './data/embed/std_nature_emb.txt'
    params['double_encoder'] = 'D'  # S D
    params['data_augment'] = 'C'  # C O
    params['relu'] = 'RT'
    params['fastica'] = 'N'  # N Y
    params['r_init'] = 0.1
    params['r_upper'] = 0.5
    params['r_dropout'] = 0.4
    params['loss_mode'] = 'squareLoss'
    params['save_path'] = s_p + '_' + params['label_mode'] + '_' +params['model_name'] + '_' + params['double_encoder'] + '_' + params['data_augment'] + '_' +params['relu'] + '_' + params['fastica'] + '_' + str(params['embedding_dim']) + '_' + params['emb_name'] + '_' + params['eval_mode']

    # CNN
    params['filter_sizes'] = [2, 3, 4]

    if params['model_name'] == 'DPCNN':
        params['num_filters'] = 250
    elif params['model_name'] == 'TextCnn':
        params['num_filters'] = 128
    elif params['model_name'] == 'VDCNN':
        params['num_filters'] = 16

    # ResNet
    params['block'] = 'BasicBlock'
    params['num_blocks'] = [2, 2, 2, 2]

    # Attention
    # params['dim_model']
    params['device'] = 'cpu'
    params['hidden'] = 50
    params['last_hidden'] = 25
    params['num_head'] = 2
    params['num_encoder'] = 2

    # RNN
    params['hidden_num'] = 32   #32
    params['num_layers'] = 1
    params['num_bidirections'] = 2   #2


    params, x_train,y_train,x_test,y_test,tests,labels = Init.Init_process_2(path=file_path, params=params)


    x_train_t = torch.Tensor(x_train).long()
    y_train_t = torch.Tensor(y_train)
    x_dev_t = torch.Tensor(x_test).long()
    x_test_t = torch.Tensor(x_test).long()
    y_dev_t = torch.Tensor(y_test)
    y_test_t = torch.Tensor(y_test)
    print('train',x_train_t.shape)
    print('dev',x_dev_t.shape)
    print('test',x_test_t.shape)

    if mm == 'TextCnn':
        model_1 = TextCnn.TCM_1(params)
        print(model_1)

    elif mm == 'TextRCnn':
        model_1 = TextRCnn.TCM_1(params)
        print(model_1)

    elif mm == 'DPCNN':
        model_1 = DPCNN.TCM_1(params)
        print(model_1)


    elif mm == 'TextRnn':
        model_1 = TextRnn.TCM_1(params)
        print(model_1)



    train_datasets_1 = torch.utils.data.TensorDataset(x_train_t, y_train_t)
    train_dev_datasets = torch.utils.data.TensorDataset(x_dev_t, y_dev_t)
    test_datasets = torch.utils.data.TensorDataset(x_test_t, y_test_t)
    train_loader_1 = torch.utils.data.DataLoader(train_datasets_1, batch_size=params['batch_size'], shuffle=False)
    dev_loader = torch.utils.data.DataLoader(train_dev_datasets, batch_size=params['batch_size'], shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=params['batch_size'], shuffle=True)

    start_time, time_s = time.time(), time.ctime()
    history, model = Labels_Train.Train(model_1, params=params, train_loader=train_loader_1, dev_loader=dev_loader, test_loader=test_loader)
    # SaveEmbededPic(model, params, x_test)

    end_time, time_e = time.time(), time.ctime()
    spend_time = end_time-start_time
    print('Run Start time:{}, Run End time:{}, Run Spend time:{}'.format(time_s, time_e, spend_time))
    time_dict = {}
    time_dict['start_time'] = time_s
    time_dict['end_time'] = time_e
    time_dict['spend_time'] = spend_time
    print(spend_time)


