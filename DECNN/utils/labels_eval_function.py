from utils.headers import *
from utils.data_processor import *
from utils.check_out import Check

random.seed(100)
np.random.seed(100)
torch.manual_seed(100)
torch.cuda.manual_seed_all(100)
torch.backends.cudnn.deterministic = True


def calculate_acuracy_mode_one(model_pred, labels):

    accuracy_th = 0.5
    pred_result = model_pred > accuracy_th
    pred_result = pred_result.float()
    pred_one_num = torch.sum(pred_result)
    if pred_one_num == 0:
        return 0, 0, 0
    target_one_num = torch.sum(labels)
    true_predict_num = torch.sum(pred_result * labels)
    #
    precision = true_predict_num / pred_one_num
    recall = true_predict_num / target_one_num
    f1_score = 2*precision*recall / (precision + recall)



    return precision.numpy(), recall.numpy(), f1_score.numpy()

def calculate_acuracy_mode_two(model_pred, labels, params):
    batch_size = model_pred.size()[0]
    precision = 0
    recall = 0
    f1_score = 0
    top = params['top_k']
    pred_label_locate = torch.argsort(model_pred, descending=True)[:, 0:top]
    for i in range(model_pred.shape[0]):
        temp_label = torch.zeros(1, model_pred.shape[1])
        temp_label[0,pred_label_locate[i]] = 1
        target_one_num = torch.sum(labels[i])
        true_predict_num = torch.sum(temp_label * labels[i])
        precision += true_predict_num / top
        recall += true_predict_num / target_one_num
        f1_score += 2 * precision * recall / (precision + recall)
    return precision/batch_size, recall/batch_size, f1_score/batch_size

def AUC(pred, y_true, params):
    pred = pred.reshape((-1, len(params['label_dict'])))
    y_true =  y_true.reshape((-1, len(params['label_dict'])))

    pred1 = []
    gt1 = []
    for i in range(len(pred)):
        gx_max = np.argsort(-pred[i])[:12]
        pred1.append(pred[i][gx_max])
        gt1.append(y_true[i][gx_max])
    fpr1, tpr1, thresholds = metrics.roc_curve(np.array(gt1).reshape(-1),
                                             np.array(pred1).reshape(-1))
    plt.figure()
    plt.plot(fpr1, tpr1, marker='.')
    plt.title('Test_ROC')
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.savefig(params['save_path']+'/' + 'roc(test).png')
    # plt.show()
    plt.close()
    auc = metrics.auc(fpr1, tpr1)
    print('auc', auc)
    return auc

def evaluate_multilabel_class(params, model, data_loader,  test=False):
    model.eval()
    loss_func = params['loss_func']
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    predict_all_ = np.array([], dtype=int)
    texts_all = np.array([], dtype=int)


    sum_f1 = 0.
    sum_prec = 0.
    sum_rec = 0.

    with torch.no_grad():
        for texts, labels in data_loader:
            # print(texts)
            texts = texts.to(device)
            outputs,_, = model(texts)
            # outputs = torch.sigmoid(outputs)
            loss = loss_func(outputs, labels)
            loss_total += loss

            if params['eval_mode'] == 'ONE':
                prec, rec, f1 = calculate_acuracy_mode_one(outputs, labels)
            elif params['eval_mode'] == 'TWO':
                prec, rec, f1 = calculate_acuracy_mode_two(outputs, labels, params)

            labels = labels.data.cpu().numpy()
            labels_all = np.append(labels_all, labels)
            # predict_all = np.append(predict_all, predic)
            predict_all_ = np.append(predict_all_, outputs)
            texts_all = np.append(texts_all, texts)

            sum_prec += prec
            sum_rec += rec
            sum_f1 += f1

    ave_rec = sum_rec / len(data_loader)
    ave_prec = sum_prec / len(data_loader)
    ave_f1 = sum_f1 / len(data_loader)

    if test:
        auc = AUC(predict_all_, labels_all, params)
        return loss_total / len(data_loader), ave_prec, ave_rec, ave_f1, auc

    return loss_total / len(data_loader), ave_prec, ave_rec, ave_f1


