import torch
import model.crnn as crnn
import utils
import data.dataset as dataset
import torch.utils.data
import argparse
from warpctc_pytorch import CTCLoss
import torch.optim as optim
from utils import get_data
import os
import zhconv

global max_acc
max_acc = 0.0

def trainBatch(crnn, criterion, optimizer, train_iter, converter):
    data = next(train_iter)
    images, labels =data
    batch_size = images.size(0)
    preds = crnn(images)
    text, length = converter.encode(labels)
    predsSize = torch.IntTensor([preds.size(0)] * batch_size)
    cost = criterion(preds, text, predsSize, length) / batch_size
    crnn.zero_grad()
    cost.backward()
    optimizer.step()
    return cost

def val(net, test_dataset, criterion, epoch, args, converter):
    global max_acc
    print('Start validation')
    for para in net.parameters():
        para.requires_grad = False
    net.eval()
    dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, num_workers=0,
        shuffle=False, collate_fn=dataset.alignCollate(imgH=args.imageH, imgW=args.imageW))
    val_iter = iter(dataloader)
    max_iter = len(dataloader)

    path = './history/' + args.exp_name
    if not os.path.exists(path):
        os.mkdir(path)
    res_file = open('./history/' + args.exp_name + '/res' + str(epoch) + '.txt', 'w+')

    n_correct = 0
    loss = utils.averager()
    total = 0
    for i in range(max_iter):
        data = val_iter.next()
        images, labels = data
        batch_size = images.size(0)
        texts, lengths = converter.encode(labels)
        preds = net(images)
        preds_size = torch.IntTensor([preds.size(0)] * batch_size)
        cost = criterion(preds, texts, preds_size, lengths) / batch_size
        loss.add(cost)
        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        res_real = converter.decode(preds, preds_size, raw=False)
        for real, gt in zip(res_real, labels):
            res_file.write(str(total)+' ['+real+'] ['+gt+']\n')
            print(str(total)+' | '+real+' | '+gt+' | '+str(real==gt))
            total += 1
            real = zhconv.convert(real,'zh-cn')
            gt = zhconv.convert(gt,'zh-cn')
            if real.lower() == gt.lower():
                n_correct += 1

    print('n_correct:', n_correct)
    acc = n_correct / float(total)

    path = './history/' + args.exp_name
    if not os.path.exists(path):
        os.mkdir(path)
    f = open('./history/{}/record.txt'.format(args.exp_name), 'a+', encoding='utf-8')
    f.write("Epoch : {} | ACC : {}\n".format(epoch, acc))
    f.close()

    if acc > max_acc:
        path = './history/' + args.exp_name
        if not os.path.exists(path):
            os.mkdir(path)
        torch.save(net.state_dict(), path + '/best_model.pth')
        max_acc = acc
    print('max_acc:', max_acc)
    print('avg_loss:%f  accuracy:%f' % (loss.val(), acc))

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Batch') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--exp_name', type=str, default='test', help='')
    parser.add_argument('--batch_size', type=int, default=64, help='')
    parser.add_argument('--lr', type=float, default=1.0, help='')
    parser.add_argument('--epoch', type=int, default=1000, help='')
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--resume', type=str, default='', help='')
    parser.add_argument('--train_dataset', type=str, default='', help='')
    parser.add_argument('--test_dataset', type=str, default='', help='')
    parser.add_argument('--imageH', type=int, default=32, help='')
    parser.add_argument('--imageW', type=int, default=256, help='')
    parser.add_argument('--nh', type=int, default=256, help='')
    parser.add_argument('--alpha_path', type=str, default='./data/benchmark.txt', help='')
    args = parser.parse_args()

    alpha_file = open(args.alpha_path, 'r')
    alphabet = alpha_file.read()
    converter = utils.strLabelConverter(alphabet)

    nclass = len(alphabet) + 1
    nc = 3
    criterion = CTCLoss()
    crnn = crnn.CRNN(nc, args.nh, nclass, args.imageH)
    crnn.apply(weights_init)
    crnn = crnn.cuda()
    crnn = torch.nn.DataParallel(crnn)
    criterion = criterion.cuda()
    loss_avg = utils.averager()
    optimizer = optim.Adadelta(crnn.parameters(), lr=args.lr, rho=0.9)

    train_dataset, test_dataset = get_data(args)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True, drop_last=False,
        collate_fn=dataset.alignCollate(imgH=args.imageH, imgW=args.imageW)
    )

    if args.resume != '':
        crnn.load_state_dict(torch.load(args.resume))

    if args.test:
        val(crnn, test_dataset, criterion, 0, args, converter)
        exit()

    for epoch in range(args.epoch):
        train_iter = iter(train_dataloader)
        i = 0
        while i < len(train_dataloader):
            for para in crnn.parameters():
                para.requires_grad = True
            crnn.train()
            cost = trainBatch(crnn, criterion, optimizer, train_iter, converter)
            loss_avg.add(cost)
            i += 1
            iters = len(train_dataloader) * epoch + i
            if iters % 10 == 0:
                print('epoch:%d train:%d/%d' % (epoch, i, len(train_dataloader)), cost.item())
        val(crnn, test_dataset, criterion, epoch, args, converter)
        # # scheduler
        # if (epoch + 1) <= 40 and (epoch + 1) % 8 == 0:
        #     for p in optimizer.param_groups:
        #         p['lr'] *= 0.8
        # elif (epoch + 1) > 40 and (epoch + 1) % 2 == 0:
        #     for p in optimizer.param_groups:
        #         p['lr'] *= 0.8
