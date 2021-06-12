# original code: https://github.com/dyhan0920/PyramidNet-PyTorch/blob/master/train.py

import argparse
import os
import shutil

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import resnet as RN
import utils
import numpy as np
from tensorboardX import SummaryWriter
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Cutmix PyTorch CIFAR-10, CIFAR-100 Training')
parser.add_argument('--net_type', default='resnet', type=str,
                    help='baseline network type: resnet')
parser.add_argument('--augment_type', default='cutmix', type=str,
                    help='data augmentation method (options: cutmix, cutout, mixup, baseline)')
parser.add_argument('--dataset', dest='dataset', default='cifar10', type=str,
                    help='dataset (options: cifar10, cifar100)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.25, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--depth', default=50, type=int,
                    help='depth of the network (default: 32)')
parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false',
                    help='to use basicblock for CIFAR datasets (default: bottleneck)')
parser.set_defaults(bottleneck=True)

# parameters for mixup
parser.add_argument('--alpha', default=1., type=float,
                    help='mixup interpolation coefficient (default: 1)')

# parameters for cutmix
parser.add_argument('--beta', default=1., type=float,
                    help='hyperparameter beta')
parser.add_argument('--cutmix_prob', default=1., type=float,
                    help='cutmix probability')

# parameters for cutout
parser.add_argument('--n_holes', type=int, default=1,
                    help='number of holes to cut out from image')
parser.add_argument('--length', type=int, default=16,
                    help='length of the holes')


def main():
    global args
    args = parser.parse_args()
    print(args)

    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    # choose dataset: cifar10 or cifar100
    if args.dataset == 'cifar100':
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('../data', train=True, download=True, transform=transform_train),
            batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('../data', train=False, transform=transform_test),
            batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
        numberofclass = 100
    elif args.dataset == 'cifar10':
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('../data', train=True, download=True, transform=transform_train),
            batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('../data', train=False, transform=transform_test),
            batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
        numberofclass = 10
    else:
        raise Exception('unknown dataset: {}'.format(args.dataset))

    print("=> creating model '{}'".format(args.net_type))

    model = RN.ResNet(args.depth, numberofclass, args.bottleneck)  # for ResNet

    model = torch.nn.DataParallel(model).cuda()

    print('the number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    # for baseline, cutout and cutmix; for mixup we should use utils.mixup_criterion(criterion, pred, y_a, y_b, lam)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay, nesterov=True)

    cudnn.benchmark = True

    # tensorboard: create writer
    sumwriter = SummaryWriter('./runs/'+args.augment_type)

    val_loss_best = 100.
    val_acc_best = 0.

    for epoch in range(0, args.epochs):

        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train_loss, train_accuracy = train(train_loader, model, sumwriter, criterion, optimizer, epoch, args.batch_size, args.augment_type)

        # evaluate on validation set
        val_loss, val_accuracy = validate(val_loader, model, criterion, epoch)

        ###### tensorboardX
        sumwriter.add_scalar("train_loss_per_epoch", train_loss, epoch)
        sumwriter.add_scalar('train_accuracy_per_epoch', train_accuracy, epoch)
        sumwriter.add_scalar("val_loss_per_epoch", val_loss, epoch)
        sumwriter.add_scalar('val_accuracy_per_epoch', val_accuracy, epoch)
        ######

        # remember best accurracy and save checkpoint
        is_best = val_accuracy >= val_acc_best
        val_acc_best = max(val_accuracy, val_acc_best)

        # print('Current best accuracy (top-1 and 5 error):', best_err1, best_err5)
        save_checkpoint({
            'epoch': epoch,
            'net_type': args.net_type,
            'augment_type':args.augment_type,
            'state_dict': model.state_dict(),
            'accuracy': val_acc_best,
            'optimizer': optimizer.state_dict(),
        }, is_best)

    print('Best accuracy:', val_acc_best)


def train(train_loader, model, sumwriter, criterion, optimizer, epoch, batch_size, augment_type):
    """
    # train for one epoch

    Args:
        train_loader:
        model:
        sumwriter: tensorboardX.SummaryWriter
        criterion:
        optimizer:
        epoch:
        augment_type: data augmentation method (options: cutmix, cutout, mixup, baseline)

    Returns:

    """
    n_train = len(train_loader) # n_train is the number of batches

    # switch to train mode
    model.train()
    if augment_type == 'baseline':
        train_loss = 0
        correct = 0
        total = 0
        # positive_num = 0    # 正样本个数
        with tqdm(total=n_train, desc=f'EpochTrain {epoch + 1}/{args.epochs}', unit='batch') as pbar:
            for ii, (input, target) in enumerate(train_loader):

                # no data augmentation
                input, target = input.cuda(), target.cuda()

                # compute output
                output = model(input)

                # compute loss
                loss = criterion(output, target)
                train_loss += loss.item()

                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Calculate running average of accuracy
                pred = torch.max(output.data, 1)[1]
                total += target.size(0)
                # positive_num += target.data.sum()
                correct += (pred == target.data).sum().item()
                accuracy = correct / total

                # print and tensorboard loss and accuracy
                sumwriter.add_scalar('train_loss_per_iteration', loss.item(), ii + epoch * (1 + n_train))
                sumwriter.add_scalar('average_accuracy_per_iteration', accuracy, ii + epoch * (1 + n_train))
                pbar.set_postfix(**{'loss(batch)': loss.item(), 'accuracy': accuracy})
                pbar.update(1)
        return train_loss/(ii+1), accuracy

    elif augment_type == 'cutmix':
        train_loss = 0
        correct = 0
        total = 0
        # current_LR = get_learning_rate(optimizer)[0]
        with tqdm(total=n_train, desc=f'EpochTrain {epoch + 1}/{args.epochs}', unit='batch') as pbar:
            for ii, (input, target) in enumerate(train_loader):
                # measure data loading time
                # data_time.update(time.time() - end)

                input, target = input.cuda(), target.cuda()

                # data augmentation
                if args.beta > 0:
                    # generate mixed sample
                    lam = np.random.beta(args.beta, args.beta)
                    rand_index = torch.randperm(input.size()[0]).cuda()
                    target_a = target
                    target_b = target[rand_index]
                    bbx1, bby1, bbx2, bby2 = utils.rand_bbox(input.size(), lam)
                    input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
                    # adjust lambda to exactly match pixel ratio
                    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
                    # compute output
                    output = model(input)
                    loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)
                    train_loss += loss
                else:
                    raise Exception('For cutmix, the beta({}) should be bigger than 0'.format(args.beta))

                # Calculate running average of accuracy
                # _, predicted = torch.max(output.data, 1)
                # total += target.size(0)
                # correct += (lam * predicted.eq(target_a.data).cpu().sum().float()
                #             + (1 - lam) * predicted.eq(target_b.data).cpu().sum().float())
                # accuracy = correct/total
                pred = torch.max(output.data, 1)[1]
                total += target.size(0)
                # positive_num += target.data.sum()
                correct += (pred == target.data).sum().item()
                accuracy = correct / total

                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # print and tensorboard loss and accuracy
                sumwriter.add_scalar('train_loss_per_iteration', loss.item(), ii + epoch * (1 + n_train))
                sumwriter.add_scalar('average_accuracy_per_iteration', accuracy, ii + epoch * (1 + n_train))
                pbar.set_postfix(**{'loss(batch)': loss.item(), 'accuracy': accuracy})
                pbar.update(1)
        return train_loss/(ii+1), accuracy

    elif augment_type == 'mixup':
        train_loss = 0
        reg_loss = 0
        correct = 0
        total = 0
        with tqdm(total=n_train, desc=f'EpochTrain {epoch + 1}/{args.epochs}', unit='batch') as pbar:
            for ii, (input, target) in enumerate(train_loader):

                input, target = input.cuda(), target.cuda()

                # data augmentation
                input, target_a, target_b, lam = utils.mixup_data(input, target,
                                                               args.alpha)
                input, target_a, target_b = map(Variable, (input,
                                                              target_a, target_b))
                # compute output
                output = model(input)
                # compute loss
                loss = utils.mixup_criterion(criterion, output, target_a, target_b, lam)
                train_loss += loss.item()

                # Calculate running average of accuracy
                # _, predicted = torch.max(output.data, 1)
                # total += target.size(0)
                # correct += (lam * predicted.eq(target_a.data).cpu().sum().float()
                #             + (1 - lam) * predicted.eq(target_b.data).cpu().sum().float())
                # accuracy = correct/total
                pred = torch.max(output.data, 1)[1]
                total += target.size(0)
                # positive_num += target.data.sum()
                correct += (pred == target.data).sum().item()
                accuracy = correct / total

                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # print and tensorboard loss and accuracy
                sumwriter.add_scalar('train_loss_per_iteration', loss.item(), ii + epoch * (1 + n_train))
                sumwriter.add_scalar('average_accuracy_per_iteration', accuracy, ii + epoch * (1 + n_train))
                pbar.set_postfix(**{'loss(batch)': loss.item(), 'accuracy': accuracy})
                pbar.update(1)
        return train_loss/(ii+1), accuracy

    elif augment_type == 'cutout':
        train_loss = 0
        correct = 0
        total = 0
        with tqdm(total=n_train, desc=f'EpochTrain {epoch + 1}/{args.epochs}', unit='batch') as pbar:
            for ii, (input, target) in enumerate(train_loader):

                # data augmentation
                input = utils.cutout(input, args.n_holes, args.length)

                input, target = input.cuda(), target.cuda()

                # compute output
                output = model(input)

                # compute loss
                loss = criterion(output, target)
                train_loss += loss.item()

                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Calculate running average of accuracy
                pred = torch.max(output.data, 1)[1]
                total += target.size(0)
                correct += (pred == target.data).sum().item()
                accuracy = correct / total

                # print and tensorboard loss and accuracy
                sumwriter.add_scalar('train_loss_per_iteration', loss.item(), ii + epoch * (1 + n_train))
                sumwriter.add_scalar('average_accuracy_per_iteration', accuracy, ii + epoch * (1 + n_train))
                pbar.set_postfix(**{'loss(batch)': loss.item(), 'accuracy': accuracy})
                pbar.update(1)
        return train_loss/(ii+1), accuracy

    else:
        raise Exception('unknown augment_type: {}'.format(args.augment_type))


def validate(val_loader, model, criterion, epoch):
    n_val = len(val_loader)
    val_loss = 0
    correct = 0
    total = 0

    # switch to evaluate mode
    model.eval()

    with tqdm(total=n_val, desc=f'EpochValidation{epoch+1}/{args.epochs}', unit='batch') as pbar:
        for ii, (input, target) in enumerate(val_loader):
            input, target = input.cuda(), target.cuda()

            output = model(input)

            loss = criterion(output, target)
            val_loss += loss.item()

            # Calculate running average of accuracy
            pred = torch.max(output.data, 1)[1]
            total += target.size(0)
            correct += (pred == target.data).sum().item()
            accuracy = correct / total
            # print loss and accuracy
            pbar.set_postfix(**{'loss(batch)': loss.item(), 'accuracy': accuracy})
            pbar.update(1)
    return val_loss/(ii+1), accuracy


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    directory = "runs/%s/" % (args.augment_type)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    # torch.save(state, filename)
    if is_best:
        torch.save(state, filename)
        shutil.copyfile(filename, 'runs/%s/' % (args.augment_type) + 'model_best.pth.tar')


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // (args.epochs * 0.5))) * (0.1 ** (epoch // (args.epochs * 0.75)))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr


if __name__ == '__main__':
    main()
