from utils.headers import *
from utils.classes_eval_function import *
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
    dev_best_acc = 0.0
    flag = False         # 记录是否很久没有效果提升
    event_path = params['save_path'] + '/log'
    writer = SummaryWriter(log_dir=event_path)
    history = {}
    history['loss'] = []
    history['acc'] = []
    # history['precision'] = []
    # history['recall'] = []
    # history['f1'] = []

    history['val_loss'] = []
    history['val_acc'] = []
    history['val_precision'] = []
    history['val_recall'] = []
    history['val_f1'] = []
    for epoch in range(params['epochs']):
        lr = scheduler.get_last_lr()[0]
        temp_train_loss = 0.0
        temp_train_acc = 0.0
        temp_dev_loss = 0.0
        temp_dev_acc = 0.0
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
            loss = loss_func(preds, y)
            loss.backward()
            optimizer.step()

            # 检查点
            if total_batch % params['ep'] == 0:
            # if iter_count % params['ep'] == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = y.data.cpu()
                predic = torch.max(preds.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss, dev_prec, dev_recall, dev_f1 = evaluate_multi_class(params, model, dev_loader)

                temp_train_loss += loss.item()
                temp_train_acc += train_acc
                temp_dev_loss += dev_loss
                temp_dev_acc += dev_acc
                temp_dev_prec += dev_prec
                temp_dev_recall += dev_recall
                temp_dev_f1 += dev_f1

                if dev_acc > dev_best_acc:
                    dev_best_acc = dev_acc
                    torch.save(model.state_dict(), params['save_path'] + '/' + params['model_name'] + '.ckpt')
                    # model.state_dict()保存的是模型的框架的参数名以及参数
                    improve = '*'
                    best_time = time.time()
                    params['spend_time'] = best_time - start_time
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Train Loss: {1:>5.2}, Train Acc: {2:>6.2%} '\
                      '              Val Loss: {3:>5.2}, Val Acc: {4:>6.2%},Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc,time_dif, improve))
                # print(msg.format(iter_count, loss.item(), train_acc, dev_loss, dev_acc,time_dif, improve))
                writer.add_scalar('loss/train', loss.item(), total_batch)
                writer.add_scalar('loss/dev', dev_loss, total_batch)
                writer.add_scalar('acc/train', train_acc, total_batch)
                writer.add_scalar('acc/dev', dev_acc, total_batch)
                writer.add_scalar('precision/dev', dev_prec, total_batch)
                writer.add_scalar('recall/dev', dev_recall, total_batch)
                writer.add_scalar('f1/dev', dev_f1, total_batch)
                writer.add_hparams(hparam_dict={'lr':params['learning_rate'], 'bsize':params['batch_size']},metric_dict={'train_loss':loss.item(), 'train_acc':train_acc, 'dev_loss':dev_loss, 'dev_acc':dev_acc})
            iter_count += 1
            total_batch += 1
        train_acc = temp_train_acc / iter_count
        train_loss = temp_train_loss / iter_count
        dev_acc = temp_dev_acc / iter_count
        dev_loss = temp_dev_loss / iter_count
        dev_prec = temp_dev_prec/ iter_count
        dev_recall = temp_dev_recall / iter_count
        dev_f1 = temp_dev_f1 / iter_count

        scheduler.step()
        if flag:
            break
        history['acc'].append(train_acc)
        history['loss'].append(train_loss)
        history['val_acc'].append(dev_acc)
        history['val_loss'].append(dev_loss)
        history['val_precision'].append(dev_prec)
        history['val_recall'].append(dev_recall)
        history['val_f1'].append(dev_f1)
        for name, parameters in model.named_parameters():
            if name == 'embedding_g':
                embeds = parameters
                metadata = list(params['ind2word'].keys())
                metadata = torch.from_numpy(np.array(metadata, 'int32'))
                writer.add_embedding(embeds, metadata=metadata)
                writer.close()
                embeds = embeds.data.numpy()
                labels = list(params['ind2word'].values())
                with open(params['save_path'] + '/embed_g.txt', 'w', encoding='utf-8') as f:
                    for i in range(len(params['ind2word']) - 1):
                        temp_emb = embeds[i]
                        f.write(str(i) + '----' + labels[i] + '----' + str(temp_emb))
                        f.write('\n')

            elif name == 'embedding_n':
                embeds = parameters
                metadata = list(params['ind2word'].keys())
                metadata = torch.from_numpy(np.array(metadata, 'int32'))
                writer.add_embedding(embeds, metadata=metadata)
                writer.close()
                embeds = embeds.data.numpy()
                labels = list(params['ind2word'].values())
                with open(params['save_path'] + '/embed_n.txt', 'w', encoding='utf-8') as f:
                    for i in range(len(params['ind2word']) - 1):
                        temp_emb = embeds[i]
                        f.write(str(i) + '----' + labels[i] + '----' + str(temp_emb))
                        f.write('\n')
    test(params, model, test_loader)    # 最后了~
    saveFile(params)
    words = params['text_words']
    SaveEmbededPic(model, params, words)
    CalcuSimi(params, words)
    return history, model


def test(params, model, test_loader):
    model.load_state_dict(torch.load(params['save_path'] + '/' + params['model_name'] + '.ckpt'))
    model.eval()
    start_time = time.time()

    # test_acc, test_loss, test_precision, test_recall, test_f1,  test_report, test_confusion= eval_func(params, model, test_loader, test=True)
    test_acc, test_loss, test_report, test_save_report, test_confusion= evaluate_multi_class(params, model, test_loader, test=True)
    msg = 'Test Loss: {0:>5.2}, Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print('Precision, Recall and F1-Score...')
    print(test_report)
    print('Confusion Matrix...')
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print('Time usage', time_dif)
    df = pd.DataFrame(test_save_report)
    df = pd.DataFrame(df.T, index=df.columns, columns=df.index)
    df.to_excel(params['save_path'] + '/' + 'test_report.xlsx')



