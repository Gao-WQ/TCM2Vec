from utils.headers import *
from utils.labels_eval_function import *
from utils.Save import *

random.seed(100)
np.random.seed(100)
torch.manual_seed(100)
torch.cuda.manual_seed_all(100)
torch.backends.cudnn.deterministic = True

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def Train(model, params, train_loader, test_loader, dev_loader):
    start_time = time.time()
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
    kwargs = {'step_size': params['step_size'], 'gamma': params['gamma']}
    scheduler = lr_scheduler.StepLR(optimizer, **kwargs)

    loss_func = params['loss_func']
    total_batch = 0
    dev_best_f1 = 0.0
    flag = False         # 记录是否很久没有效果提升
    event_path = params['save_path'] + '/log'
    writer = SummaryWriter(log_dir=event_path)
    history = {}
    history['loss'] = []
    history['train_precision'] = []
    history['train_recall'] = []
    history['train_f1'] = []

    history['val_loss'] = []
    history['val_precision'] = []
    history['val_recall'] = []
    history['val_f1'] = []

    for epoch in range(params['epochs']):
        lr = scheduler.get_last_lr()[0]
        temp_train_loss = 0.0
        temp_train_prec = 0.0
        temp_train_recall = 0.0
        temp_train_f1 = 0.0

        temp_dev_loss = 0.0
        temp_dev_prec = 0.0
        temp_dev_recall = 0.0
        temp_dev_f1 = 0.0

        # print('Epoch [{}/{}]'.format(epoch + 1, params['epochs']))
        print('Epoch [{}/{}], lr: {}'.format(epoch + 1, params['epochs'], lr))

        iter_count = 0 # 记录每个epoch中保存训练数据的迭代次数
        for i, (x, y) in enumerate(train_loader):
            x = Variable(x)
            y = Variable(y)
            x = x.to(device)
            y = y.to(device)
            preds,_ = model(x)
            model.zero_grad()
            # loss = loss_func(torch.sigmoid(preds), y)
            loss = loss_func(preds, y)
            loss.backward()
            optimizer.step()

            # 检查点
            if total_batch % params['ep'] == 0:
            # if iter_count % params['ep'] == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = y.data.cpu()
                # predic = torch.max(preds.data, 1)[1].cpu()
                if params['eval_mode'] == 'ONE':
                    train_prec, train_rec, train_f1 = calculate_acuracy_mode_one(torch.sigmoid(preds), y)
                elif params['eval_mode'] == 'TWO':
                    train_prec, train_rec, train_f1 = calculate_acuracy_mode_two(torch.sigmoid(preds), y, params)


                ave_loss, ave_prec, ave_rec, ave_f1 = evaluate_multilabel_class(params, model, dev_loader)

                temp_train_loss += loss.item()
                temp_train_prec += train_prec
                temp_train_recall += train_rec
                temp_train_f1 += train_f1

                temp_dev_loss += ave_loss
                temp_dev_prec += ave_prec
                temp_dev_recall += ave_rec
                temp_dev_f1 += ave_f1

                if temp_dev_f1 > dev_best_f1:
                    dev_best_f1 = temp_dev_f1
                    torch.save(model.state_dict(), params['save_path'] + '/' + params['model_name'] + '.ckpt')
                    # model.state_dict()保存的是模型的框架的参数名以及参数
                    improve = '*'
                    best_time = time.time()
                    params['spend_time'] = best_time - start_time
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Train Loss: {1:>5.2}, Train F1: {2:>6.2%} '\
                      '              Val Loss: {3:>5.2}, Val F1: {4:>6.2%}, Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_f1, ave_loss, ave_f1 ,time_dif, improve))
                # print(msg.format(iter_count, loss.item(), train_acc, dev_loss, dev_acc,time_dif, improve))
                writer.add_scalar('loss/train', loss.item(), total_batch)
                writer.add_scalar('loss/dev', ave_loss, total_batch)
                writer.add_scalar('prec/train', train_prec, total_batch)
                writer.add_scalar('rec/train', train_rec, total_batch)
                writer.add_scalar('f1/train', train_f1, total_batch)

                writer.add_scalar('prec/dev', ave_prec, total_batch)
                writer.add_scalar('rec/dev', ave_rec, total_batch)
                writer.add_scalar('f1/dev', ave_f1, total_batch)

                writer.add_hparams(hparam_dict={'lr':params['learning_rate'], 'bsize':params['batch_size']},metric_dict={'train_loss':loss.item(), 'train_prec':train_prec, 'train_rec':train_rec, 'train_f1':train_f1,
                                                                                                                         'dev_loss':ave_loss,  'dev_prec':ave_prec, 'dev_rec':ave_rec, 'dev_f1':ave_f1,})
            iter_count += 1
            total_batch += 1

        train_loss = temp_train_loss / iter_count
        train_prec = temp_train_prec/ iter_count
        train_recall = temp_train_recall / iter_count
        train_f1 = temp_train_f1 / iter_count

        dev_loss = temp_dev_loss / iter_count
        dev_prec = temp_dev_prec/ iter_count
        dev_recall = temp_dev_recall / iter_count
        dev_f1 = temp_dev_f1 / iter_count

        scheduler.step()
        if flag:
            break

        history['loss'].append(train_loss)
        history['train_precision'].append(train_prec)
        history['train_recall'].append(train_recall)
        history['train_f1'].append(train_f1)

        history['val_loss'].append(dev_loss)
        history['val_precision'].append(dev_prec)
        history['val_recall'].append(dev_recall)
        history['val_f1'].append(dev_f1)


    test(params, model, test_loader)    # 最后了~
    saveFile_2(params)
    words = params['text_words']
    SaveEmbededPic(model, params, words)
    CalcuSimi(params, words)
    return history, model


def test(params, model, test_loader):
    model.load_state_dict(torch.load(params['save_path'] + '/' + params['model_name'] + '.ckpt'))
    model.eval()
    start_time = time.time()

    ave_loss, ave_pre, ave_rec, ave_f1, auc = evaluate_multilabel_class(params, model, test_loader, test=True)
    result = {'loss': ave_loss.item(), 'auc':auc, 'prec': ave_pre, 'rec': ave_rec, 'f1': ave_f1}
    data = pd.DataFrame(result, index=[0])
    data.to_excel(params['save_path'] + '/' + 'test_result.xlsx')
    msg = 'Test Loss: {0:>5.2}, Test f1: {1:>6.2%}'
    print(msg.format(ave_loss, ave_f1))
    time_dif = get_time_dif(start_time)
    print('Time usage', time_dif)




