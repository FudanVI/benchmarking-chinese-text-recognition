import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
from model.transocr import Transformer
from utils import get_data_package, converter, tensor2str, get_alphabet
import zhconv

parser = argparse.ArgumentParser(description='')
parser.add_argument('--exp_name', type=str, default='test', help='')
parser.add_argument('--batch_size', type=int, default=32, help='')
parser.add_argument('--lr', type=float, default=1.0, help='')
parser.add_argument('--epoch', type=int, default=1000, help='')
parser.add_argument('--radical', action='store_true', default=False)
parser.add_argument('--test', action='store_true', default=False)
parser.add_argument('--resume', type=str, default='', help='')
parser.add_argument('--train_dataset', type=str, default='', help='')
parser.add_argument('--test_dataset', type=str, default='', help='')
parser.add_argument('--imageH', type=int, default=32, help='')
parser.add_argument('--imageW', type=int, default=256, help='')
parser.add_argument('--coeff', type=float, default=1.0, help='')
parser.add_argument('--alpha_path', type=str, default='./data/benchmark.txt', help='')
parser.add_argument('--alpha_path_radical', type=str, default='./data/radicals.txt', help='')
parser.add_argument('--decompose_path', type=str, default='./data/decompose.txt', help='')
args = parser.parse_args()

alphabet = get_alphabet(args, 'char')
print('alphabet:',alphabet)

model = Transformer(args).cuda()
model = nn.DataParallel(model)
train_loader, test_loader = get_data_package(args)
optimizer = optim.Adadelta(model.parameters(), lr=args.lr, rho=0.9, weight_decay=1e-4)
criterion = torch.nn.CrossEntropyLoss().cuda()
best_acc = -1

if args.resume.strip() != '':
    model.load_state_dict(torch.load(args.resume))
    print('loading pretrained model！！！')

def train(epoch, iteration, image, length, text_input, text_gt, length_radical, radical_input, radical_gt):
    model.train()
    optimizer.zero_grad()
    result = model(image, length, text_input, length_radical, radical_input)

    text_pred = result['pred']
    loss_char = criterion(text_pred, text_gt)
    if args.radical:
        radical_pred = result['radical_pred']
        loss_radical = criterion(radical_pred, radical_gt)
        loss = loss_char + args.coeff * loss_radical
        print(
            'epoch : {} | iter : {}/{} | loss : {} | char : {} | radical : {} '.format(epoch, iteration, len(train_loader), loss, loss_char, loss_radical))

    else:
        loss = loss_char
        print('epoch : {} | iter : {}/{} | loss : {}'.format(epoch, iteration, len(train_loader), loss))
    loss.backward()
    optimizer.step()

test_time = 0
@torch.no_grad()
def test(epoch):

    torch.cuda.empty_cache()
    global test_time
    test_time += 1
    torch.save(model.state_dict(), './history/{}/model.pth'.format(args.exp_name))
    result_file = open('./history/{}/result_file_test_{}.txt'.format(args.exp_name, test_time), 'w+', encoding='utf-8')

    print("Start Eval!")
    model.eval()
    dataloader = iter(test_loader)
    test_loader_len = len(test_loader)

    correct = 0
    total = 0

    for iteration in range(test_loader_len):
        data = dataloader.next()
        image, label, _ = data
        image = torch.nn.functional.interpolate(image, size=(args.imageH, args.imageW))
        length, text_input, text_gt, length_radical, radical_input, radical_gt, string_label = converter(label, args)
        max_length = max(length)

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
        text_prob_list = []
        for i in range(batch):
            now_pred = []
            for j in range(max_length):
                if pred[i][j] != len(alphabet) - 1:
                    now_pred.append(pred[i][j])
                else:
                    break
            text_pred_list.append(torch.Tensor(now_pred)[1:].long().cuda())

            overall_prob = 1.0
            for j in range(len(now_pred)):
                overall_prob *= prob[i][j]
            text_prob_list.append(overall_prob)

        start = 0
        for i in range(batch):
            state = False
            pred = zhconv.convert(tensor2str(text_pred_list[i], args),'zh-cn')
            gt = zhconv.convert(tensor2str(text_gt_list[i], args), 'zh-cn')

            if pred == gt:
                correct += 1
                state = True
            start += i
            total += 1
            print('{} | {} | {} | {} | {} | {}'.format(total, pred, gt, state, text_prob_list[i],
                                                            correct / total))
            result_file.write(
                '{} | {} | {} | {} | {} \n'.format(total, pred, gt, state, text_prob_list[i]))


    print("ACC : {}".format(correct/total))
    global best_acc
    if correct/total > best_acc:
        best_acc = correct / total
        torch.save(model.state_dict(), './history/{}/best_model.pth'.format(args.exp_name))

    f = open('./history/{}/record.txt'.format(args.exp_name),'a+',encoding='utf-8')
    f.write("Epoch : {} | ACC : {}\n".format(epoch, correct/total))
    f.close()

if __name__ == '__main__':
    print('-------------')
    if not os.path.isdir('./history/{}'.format(args.exp_name)):
        os.mkdir('./history/{}'.format(args.exp_name))
    if args.test:
        test(-1)
        exit(0)

    for epoch in range(args.epoch):
        torch.save(model.state_dict(), './history/{}/model.pth'.format(args.exp_name))
        dataloader = iter(train_loader)
        train_loader_len = len(train_loader)
        print('length of training datasets:', train_loader_len)
        for iteration in range(train_loader_len):
            data = dataloader.next()
            image, label, _ = data
            image = torch.nn.functional.interpolate(image, size=(args.imageH, args.imageW))

            length, text_input, text_gt, length_radical, radical_input, radical_gt, string_label = converter(label, args)
            train(epoch, iteration, image, length, text_input, text_gt, length_radical, radical_input, radical_gt)

        test(epoch)

        # # scheduler
        # if (epoch + 1) <= 40 and (epoch + 1) % 8 == 0:
        #     for p in optimizer.param_groups:
        #         p['lr'] *= 0.8
        # elif (epoch + 1) > 40 and (epoch + 1) % 2 == 0:
        #     for p in optimizer.param_groups:
        #         p['lr'] *= 0.8