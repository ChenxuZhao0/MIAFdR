import logging
import os
import datetime

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import numpy as np

from torch.utils.data import DataLoader
from torch import optim

from utils.parser import get_parameter
from utils.dataset import construct_datasets
from utils.model import ResNet18, ResNet50, VGG16TI


def train_binary(args, logging, model, loss_func, optimizer, train_loader, epochs=200, lr=0.001, scheduler=None):

    for epoch in range(epochs):
        model.train()
        running_loss, n_batches, total, correct = 0.0, 0, 0, 0

        for idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(args.device), labels.to(args.device)
            outputs = model(images)
            loss = loss_func(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()
            n_batches += 1

        if scheduler is not None:
            scheduler.step()

        loss = running_loss / n_batches
        accuracy = 100 * correct / total
        print('Epoch %d training loss: %.3f training accuracy: %.3f%%' % (epoch, loss, accuracy))
        logging.info('Epoch %d training loss: %.3f training accuracy: %.3f%%' % (epoch, loss, accuracy))


def train(args, logging, model, loss_func, optimizer, train_loader, scheduler=None):
    for epoch in range(args.epoch):
        model.train()
        running_loss, n_batches, total, correct = 0.0, 0, 0, 0

        for idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(args.device), labels.to(args.device)
            outputs = model(images)
            loss = loss_func(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()
            n_batches += 1

        if scheduler is not None:
            scheduler.step()

        loss = running_loss / n_batches
        accuracy = 100 * correct / total
        print('Epoch %d training loss: %.3f training accuracy: %.3f%%' % (epoch, loss, accuracy))
        logging.info('Epoch %d training loss: %.3f training accuracy: %.3f%%' % (epoch, loss, accuracy))


def test(args, logging, model, test_loader):
    model.eval()
    total, correct = 0, 0

    with torch.no_grad():
        for idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(args.device), labels.to(args.device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Test accuracy: %.2f%%' % (100 * correct / total))
    logging.info('Test accuracy: %.2f%%' % (100 * correct / total))

    return correct / total


def evaluate(args, logging, model, test_loader):
    model.eval()
    labels_all, preds_all, probs_all = [], [], []

    with torch.no_grad():
        for idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(args.device), labels.to(args.device)
            outputs = model(images)
            _, preds = torch.max(outputs.data, 1)
            probs = F.softmax(outputs, dim=1)[:, 1]
            labels_all.append(labels)
            preds_all.append(preds)
            probs_all.append(probs)

    labels_all = torch.cat(labels_all, 0).cpu().data.numpy()
    preds_all = torch.cat(preds_all, 0).cpu().data.numpy()
    probs_all = torch.cat(probs_all, 0).cpu().data.numpy()
    return labels_all, preds_all, probs_all


def main():
    args = get_parameter()

    """# Create a log file"""
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)

    log_path = args.dataset + '-pretraining-log-%s' % (datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S"))
    log_path = log_path + '.txt'
    logging.basicConfig(
        filename=os.path.join(args.logdir, log_path),
        format="%(asctime)s - %(name)s - %(message)s",
        datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO, filemode='w')

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.info(str(args))

    if torch.cuda.is_available():
        #args.device = torch.device("cuda")
        cudnn.benchmark = True
    else:
        args.device = "cpu"
    print(f'device: {args.device}')
    logger.info(f'device: {args.device}')

    if not os.path.exists(args.moddir):
        os.makedirs(args.moddir)

    train_target_data, train_source_data, test_target_data, test_source_data = construct_datasets(args, args.dataset, args.datadir)
    train_target_loader = DataLoader(train_target_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_target_loader = DataLoader(test_target_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    train_source_loader = DataLoader(train_source_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_source_loader = DataLoader(test_source_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    print(f'training target data: {len(train_target_data)}, test target data: {len(test_target_data)}, train source data: {len(train_source_data)}, test source data: {len(test_source_data)}')

    if args.net == 'ResNet18':
        target_model = ResNet18(num_classes=args.num_classes).to(args.device)
        source_model = ResNet18(num_classes=args.num_classes).to(args.device)
    elif args.net == 'ResNet50':
        target_model = ResNet50(num_classes=args.num_classes).to(args.device)
        source_model = ResNet50(num_classes=args.num_classes).to(args.device)
    elif args.net == 'VGG16TI':
        target_model = VGG16TI(num_classes=args.num_classes).to(args.device)
        source_model = VGG16TI(num_classes=args.num_classes).to(args.device)
    else:
        raise NotImplementedError("Not support!")

    loss_func = nn.CrossEntropyLoss()
    if args.opt == 'Adam':
        target_optimizer = optim.Adam(target_model.parameters(), lr=args.lr)
        source_optimizer = optim.Adam(source_model.parameters(), lr=args.lr)
    elif args.opt == 'SGD':
        target_optimizer = optim.SGD(target_model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        source_optimizer = optim.SGD(source_model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    else:
        raise NotImplementedError("Not support!")

    target_scheduler = optim.lr_scheduler.MultiStepLR(target_optimizer, [int(args.epoch * 3 / 8), int(args.epoch * 5 / 8), int(args.epoch * 7 / 8)])
    source_scheduler = optim.lr_scheduler.MultiStepLR(source_optimizer, [int(args.epoch * 3 / 8), int(args.epoch * 5 / 8), int(args.epoch * 7 / 8)])

    if not os.path.exists(args.moddir):
        os.makedirs(args.moddir)

    train(args, logging, target_model, loss_func, target_optimizer, train_target_loader, target_scheduler)
    test(args, logging, target_model, test_target_loader)

    state = {
        'net': target_model.state_dict(),
        'epoch': args.epoch,
        'batch_size': args.batch_size
    }
    torch.save(state, os.path.join(args.moddir, args.dataset + '_' + str(args.train_size) + '_' + args.net + '_target_model.pth'))

    train(args, logging, source_model, loss_func, source_optimizer, train_source_loader, source_scheduler)
    test(args, logging, source_model, test_source_loader)

    state = {
        'net': source_model.state_dict(),
        'epoch': args.epoch,
        'batch_size': args.batch_size
    }
    torch.save(state, os.path.join(args.moddir, args.dataset + '_' + str(args.train_size) + '_' + args.net + '_source_model.pth'))



if __name__ == '__main__':
    main()
    


    