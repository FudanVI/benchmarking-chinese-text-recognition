import torch
import torch.nn as nn
import torch.optim as optim
import datetime
import warnings
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter

from args import args
from model.TransformerSTR import Transformer
from util import get_data_package, converter, tensor2str, \
    saver, get_alphabet, equal

#-------ignore the warning information-------
warnings.filterwarnings("ignore")

#---------preparation-----------
writer = SummaryWriter('runs/{}'.format(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')))
#save python files
saver(args)
#get the alphabet
alphabet = get_alphabet(args)

train_loader, test_loader = get_data_package(args)

model = Transformer(args).cuda()
model = nn.DataParallel(model)

optimizer = optim.Adadelta(model.parameters(), lr=args.lr, rho=0.9, weight_decay=1e-4)
criterion = torch.nn.CrossEntropyLoss().cuda()

#--------load pretrain model-------
if args.resume.strip() != '':
    print('loading model..')
    model.load_state_dict(torch.load(args.resume))

#---------model training-----------
times = 0 # the currant training iteration
def train(epoch, iteration, image, length, text_input, text_gt):
    global times
    model.train()
    optimizer.zero_grad()

    result = model(image, length, text_input)
    text_pred = result['pred']

    loss = criterion(text_pred, text_gt)
    loss.backward()
    optimizer.step()

    # print('epoch : {} | iter : {}/{} | loss : {}'.format(epoch, iteration, len(train_loader), loss))
    writer.add_scalar('loss', loss, times)
    times += 1

    return loss.item()


# #---------model testing----------
test_times = 0
best_acc = -1
@torch.no_grad()
def test(epoch):
    print('start validation!')
    torch.cuda.empty_cache()

    global test_times
    test_times += 1
    #------save the intermediate model-------
    torch.save(model.state_dict(), './history/{}/model.pth'.format(args.exp_name))
    #-------save prediction results of the currant validation
    result_file = open('./history/{}/result_file_test_{}.txt'.format(args.exp_name, test_times), 'w+', encoding='utf-8')

    model.eval()
    dataloader = iter(test_loader)
    test_loader_len = len(test_loader)
    correct = 0
    total = 0
    max_length = args.max_len

    with trange(test_loader_len) as t:
        for iteration in t:
            data = dataloader.next()
            image, label, _ = data
            length, text_input, text_gt, string_label = converter(label, args)

            batch = image.shape[0]
            pred = torch.zeros(batch,1).long().cuda()
            image_features = None

            prob = torch.zeros(batch, max_length).float()
            for i in range(max_length):
                length_tmp = torch.zeros(batch).long().cuda() + i + 1
                result = model(image, length_tmp, pred, conv_feature=image_features, test=True)
                prediction = result['pred']
                now_pred = torch.max(torch.softmax(prediction,2), 2)[1]
                prob[:,i] = torch.max(torch.softmax(prediction,2), 2)[0][:,-1]
                pred = torch.cat((pred, now_pred[:,-1].view(-1,1)), 1)
                image_features = result['conv']

            text_gt_list = []
            start = 0
            for i in length:
                text_gt_list.append(text_gt[start: start + i])
                start += i

            text_pred_list = []
            for i in range(batch):
                now_pred = []
                for j in range(max_length):
                    if pred[i][j] != len(alphabet) - 1:
                        now_pred.append(pred[i][j])
                    else:
                        break
                text_pred_list.append(torch.Tensor(now_pred)[1:].long().cuda())

            #---------save predictions----------
            for i in range(batch):
                state = False
                pred = tensor2str(text_pred_list[i], args)
                gt = tensor2str(text_gt_list[i], args)

                if equal(pred, gt) == 'True':
                    correct += 1
                    state = True
                total += 1
                result_file.write('{} | {} | {} | {}\n'.format(total, pred, gt, state))

            t.set_postfix(accuracy=(correct/total))

    result_file.close()
    print("ACC : {}".format(correct/total))

    #-------save the best model--------
    global best_acc
    if correct/total > best_acc:
        best_acc = correct / total
        torch.save(model.state_dict(), './history/{}/best_model.pth'.format(args.exp_name))

    #------save the validation accuracy----
    f = open('./history/{}/record.txt'.format(args.exp_name),'a+',encoding='utf-8')
    f.write("Epoch : {} | ACC : {}\n".format(epoch, correct/total))
    f.close()


if __name__ == '__main__':
    #--------only test in the testing dataset-----
    if args.test_only:
        test(-1)
        exit(0)

    for epoch in range(args.epoch):
        #-----save the model parameters of each epoch----
        torch.save(model.state_dict(), './history/{}/model.pth'.format(args.exp_name))

        dataloader = iter(train_loader)
        train_loader_len = len(train_loader)

        with trange(train_loader_len) as t:
            t.set_description("epoch:{}/{}".format(epoch, args.epoch))
            for iteration in t:
                data = dataloader.next()
                image, label, _ = data
                length, text_input, text_gt, string_label = converter(label, args)

                loss_item = train(epoch, iteration, image, length, text_input, text_gt)
                t.set_postfix(loss=loss_item)

        #---------validation--------
        if (epoch+1) % args.val_frequency == 0:
            torch.cuda.empty_cache()
            test(epoch+1)

        #---------lr schedule--------
        if (epoch+1) % args.schedule_frequency == 0:
            for p in optimizer.param_groups:
                p['lr'] *= 0.8

