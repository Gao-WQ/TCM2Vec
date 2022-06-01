from utils.headers import *
from utils.data_processor import *
from utils.check_out import Check

random.seed(100)
np.random.seed(100)
torch.manual_seed(100)
torch.cuda.manual_seed_all(100)
torch.backends.cudnn.deterministic = True

def ROC_AUC(predict_all_, labels_onehot, params):
    # 计算每一类的ROC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    labels_onehot = labels_onehot.astype('int')
    for i in range(params['num_classes']):
        fpr[i], tpr[i], _ = roc_curve(labels_onehot[:, i], predict_all_[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(labels_onehot.ravel(), predict_all_.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(params['num_classes'])]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(params['num_classes']):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= params['num_classes']
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # 保存
    with open(params['save_path'] + '/fpr.txt', 'w') as f:  # 写文件
        f.write(str(fpr))
    with open(params['save_path'] + '/tpr.txt', 'w') as f:  # 写文件
        f.write(str(tpr))
    with open(params['save_path'] + '/roc_auc.txt', 'w') as f:  # 写文件
        f.write(str(roc_auc))
    # 读取
    # with open("C:/Users/卿卿卿大爷/Desktop/文件名.txt", 'r') as f:  # 读文件
    #     line = f.read()
    #     dic = eval(line)
    # 画图
    lw = 2
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['purple', 'm', 'fuchsia', 'orchid', 'mediumvioletred', 'deeppink', 'hotpink', 'slateblue', 'darkslateblue', 'mediumslateblue', 'royalblue'])
    # for i, color in zip(range(params['num_classes']), colors):
    #     plt.plot(fpr[i], tpr[i], color=color, lw=lw,
    #              label='ROC curve of class {0} (area = {1:0.2f})'
    #                    ''.format(i, roc_auc[i]))
    for i, color in zip(range(params['num_classes']), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw)

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig(params['save_path']+'/roc.jpg')
    plt.close()
    # plt.show()


def evaluate_multi_class(params, model, data_loader,  test=False):
    model.eval()
    loss_func = nn.CrossEntropyLoss()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    predict_all_ = np.array([], dtype=int)
    texts_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_loader:
            # print(texts)
            texts = texts.to(device)
            outputs,_, = model(texts)
            loss = loss_func(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()    # 转变为tensor类型
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()         # 即使不经过softmax也是最大
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)
            predict_all_ = np.append(predict_all_, outputs)
            texts_all = np.append(texts_all, texts)

    acc = metrics.accuracy_score(labels_all, predict_all)
    predic = metrics.precision_score(labels_all, predict_all, average='macro')
    # micro方法下的precision和recall都等于accuracy。
    recall = metrics.recall_score(labels_all, predict_all, average='macro')
    f1 = metrics.f1_score(labels_all, predict_all, average='macro')

    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=params['class_list'])
        save_report = metrics.classification_report(labels_all, predict_all, target_names=params['class_list'], output_dict=True)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        Check(predict_all,labels_all,texts_all,params)

        # ROC-AUC
        predict_all_ = predict_all_.reshape((-1 ,params['num_classes']))
        labels_onehot = OneHotEncode(labels_all)
        ROC_AUC(predict_all_, labels_onehot, params)


        return acc, loss_total / len(data_loader), report, save_report, confusion

    return acc, loss_total / len(data_loader), predic, recall, f1


