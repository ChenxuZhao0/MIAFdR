import os
import time
import datetime
import pandas as pd
import numpy as np
import random as py_random
import torch
import logging
import random
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

from torchvision import datasets, transforms
from torch.utils.data import Subset, Dataset, DataLoader, random_split, ConcatDataset, TensorDataset
from statsmodels.stats.multitest import multipletests
from scipy import interpolate
from sklearn.metrics import roc_auc_score as roc_auc
from sklearn.metrics import roc_curve, accuracy_score, confusion_matrix
from statsmodels.stats.multitest import multipletests

from utils.model import ResNet18, MLPClassifier2
from utils.parser import get_parameter
from utils.dataset import CustomTransformDataset, data_random_split
from utils.utils import cal_auc, evaluate_classifier, evaluate_ours


def train_target_model(logging,pri_tra_loader,learning_rate,n_target_epochs,device,test_member_loader,test_non_member_loader):

    print('target_net Begin Training -> ')
    logging.info('target_net Begin Training -> ')
    target_net = ResNet18().to(device)

    target_criterion = nn.CrossEntropyLoss()
    target_optimizer = optim.SGD(target_net.parameters(), lr=learning_rate)
    target_scheduler = optim.lr_scheduler.MultiStepLR(target_optimizer, [n_target_epochs * 3 / 4])

    for e in range(n_target_epochs):
        # initialize gradients
        target_optimizer.zero_grad()

        running_loss, n_batches, total, correct = 0.0, 0, 0, 0

        for i, data in enumerate(pri_tra_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            target_net.train()

            # target_optimizer.zero_grad()
            outputs = target_net(inputs)
            loss = target_criterion(outputs, labels)
            target_optimizer.zero_grad()
            loss.backward()
            target_optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()
            n_batches += 1


        target_scheduler.step()

        loss = running_loss / n_batches
        accuracy = 100 * correct / total
        logging.info("uu"*50)
        logging.info('Epoch %d training loss: %.3f training accuracy: %.3f%%' % (e + 1, loss, accuracy))


        print(f'Epoch {e + 1} complete')
        logging.info(f'Epoch {e + 1} complete')

    print('target_net Finished Training')
    logging.info('target_net Finished Training')

    train_loss = []
    test_loss = []

    # evaluate the training loss and test loss
    with torch.no_grad():
        target_net.eval()
        for ind, data in enumerate(test_member_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = target_net(inputs)
            loss = target_criterion(outputs, labels)
            train_loss.append(loss.item())

    with torch.no_grad():
        target_net.eval()
        for ind, data in enumerate(test_non_member_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = target_net(inputs)
            loss = target_criterion(outputs, labels)
            test_loss.append(loss.item())

    ave_train_loss = np.mean(train_loss)
    ave_test_loss = np.mean(test_loss)
    logging.info("aa"*50)
    logging.info(f'Average training loss: {ave_train_loss}')
    logging.info(f'Average test loss: {ave_test_loss}')

    return target_net


def train_shadow_model(logging,shadow_tra_loader_list,shadow_test_loader_list,learning_rate,n_shadow_epochs,device,n_shadow=4):

    in_pred_list = []
    out_pred_list = []
    for i in range(n_shadow):
        print(f'Shadow Model {i} Begin Training -> ')
        logging.info(f'Shadow Model {i} Begin Training')

        model_i = ResNet18().to(device)

        shadow_criterion = nn.CrossEntropyLoss()
        shadow_optimizer = optim.SGD(model_i.parameters(), lr=learning_rate)
        shadow_scheduler = optim.lr_scheduler.MultiStepLR(shadow_optimizer, [n_shadow_epochs * 3 / 4])

        for e in range(n_shadow_epochs):

            shadow_optimizer.zero_grad()

            running_loss, n_batches, total, correct = 0.0, 0, 0, 0
            for ind, data in enumerate(shadow_tra_loader_list[i], 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                model_i.train()
                # shadow_optimizer.zero_grad()
                outputs = model_i(inputs)
                loss = shadow_criterion(outputs, labels)
                shadow_optimizer.zero_grad()
                loss.backward()
                shadow_optimizer.step()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                running_loss += loss.item()
                n_batches += 1

            shadow_scheduler.step()


            loss = running_loss / n_batches
            accuracy = 100 * correct / total
            logging.info('Epoch %d training loss: %.3f training accuracy: %.3f%%' % (e, loss, accuracy))

            print(f'Epoch {e + 1} complete for model {i}')
            logging.info(f'Epoch {e + 1} complete for model {i}')

        with torch.no_grad():
            model_i.eval()
            for ind, data in enumerate(shadow_tra_loader_list[i], 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model_i(inputs)
                outputs = torch.nn.functional.softmax(outputs, dim=1)
                in_pred_list.extend(outputs.cpu().numpy())

        with torch.no_grad():
            model_i.eval()
            for ind, data in enumerate(shadow_test_loader_list[i], 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model_i(inputs)
                outputs = torch.nn.functional.softmax(outputs, dim=1)
                out_pred_list.extend(outputs.cpu().numpy())

        print(f'Shadow Model {i} Finished Training')
        logging.info(f'Shadow Model {i} Finished Training')

    return in_pred_list,out_pred_list


def train_attack_classifier(args,logging,attack_hidden_layer_list, in_pred_list,out_pred_list,attack_epoch,cali_data,target_net,test_member_loader,test_non_member_loader,device,attack_learning_rate):

    print("Begin Training Attack Classifier -> ")
    logging.info("Begin Training Attack Classifier -> ")

    non_memebr_loader = DataLoader(cali_data, batch_size=32, shuffle=True)

    in_labels = torch.zeros(len(in_pred_list))
    out_labels = torch.ones(len(out_pred_list))

    # Concatenate data and labels
    in_pred_tensors = [torch.from_numpy(ndarray) for ndarray in in_pred_list]
    out_pred_tensors = [torch.from_numpy(ndarray) for ndarray in out_pred_list]

    data = torch.cat((torch.stack(in_pred_tensors), torch.stack(out_pred_tensors)))
    labels = torch.cat((in_labels, out_labels))

    logging.info(f"data shape: {data.shape}")
    logging.info(f"labels shape: {labels.shape}")

    attack_train_dataset = TensorDataset(data, labels)
    logging.info("yy"*50)
    logging.info(f"length of attack_train_dataset : {len(attack_train_dataset)}")

    attack_train_loader = DataLoader(attack_train_dataset, batch_size=32, shuffle=True)

    print("length of in_pred_list: " + str(len(in_pred_list)))
    print("length of out_pred_list : " + str(len(out_pred_list)))
    logging.info(f"length of in_pred_list : {len(in_pred_list)}")
    logging.info(f"length of out_pred_list : {len(out_pred_list)}")

    attack_model = MLPClassifier2(10, attack_hidden_layer_list, 1).to(device)

    attack_criterion = nn.BCELoss()
    attack_optimizer = optim.SGD(attack_model.parameters(), lr = attack_learning_rate)
    attack_scheduler = optim.lr_scheduler.MultiStepLR(attack_optimizer, [attack_epoch * 3 / 4])

    for epoch in range(attack_epoch):
      attack_optimizer.zero_grad()
      # attack_model.train()
      running_loss, n_batches, total, correct = 0.0, 0, 0, 0
      for inputs, labels in attack_train_loader:
          inputs, labels = inputs.to(device), labels.to(device)

          attack_model.train()

          # attack_optimizer.zero_grad()
          outputs = attack_model(inputs)
          loss = attack_criterion(outputs.squeeze(), labels)
          # labels = labels.long()
          # loss = attack_criterion(outputs, labels)
          attack_optimizer.zero_grad()
          loss.backward()
          attack_optimizer.step()

          predicted = torch.round(outputs)
          total += labels.size(0)
          correct += (predicted == labels.view(len(inputs),1)).sum().item()
          running_loss += loss.item()
          n_batches += 1

      attack_scheduler.step()

      loss = running_loss / n_batches
      accuracy = 100 * correct / total

      print("%%"*50)
      logging.info("%%"*50)
      print('Epoch %d training loss: %.3f training accuracy: %.3f%%' % (epoch, loss, accuracy))
      logging.info('Epoch %d training loss: %.3f training accuracy: %.3f%%' % (epoch, loss, accuracy))

    correct = 0
    total = 0
    with torch.no_grad():
      for inputs, labels in attack_train_loader:
          inputs, labels = inputs.to(device), labels.to(device)
          outputs = attack_model(inputs)
          predicted = torch.round(outputs)
          # _, predicted = torch.max(outputs.data, 1)
          correct += (predicted.squeeze() == labels).sum().item()
          total += labels.size(0)

    training_accuracy = correct / total
    print(f"Training accuracy of attack classifier : {training_accuracy}")
    logging.info(f"Training accuracy of attack classifier : {training_accuracy}")
    logging.info("correct over total : " + str(correct) + "  /  " + str(total))

    # evaluate testing accuracy
    Testing_correct = 0
    Testing_total = 0
    with torch.no_grad():
      for inputs, labels in non_memebr_loader:
          inputs, labels = inputs.to(device), labels.to(device)
          queries = target_net(inputs)
          queries = torch.nn.functional.softmax(queries, dim=1)
          outputs = attack_model(queries)
          predicted = torch.round(outputs)
          # _, predicted = torch.max(outputs.data, 1)
          Testing_correct += (predicted.squeeze() == labels).sum().item()
          Testing_total += labels.size(0)

    Testing_accuracy = Testing_correct / Testing_total

    logging.info("attacker test accuracy"*20)
    print(f" Testing accuracy of attack classifier : {Testing_accuracy}")
    logging.info(f"Testing accuracy of attack classifier : {Testing_accuracy}")
    logging.info("Testing correct over Testing total : " + str(Testing_correct) + "  /  " + str(Testing_total))

    non_member_cali_list = []

    with torch.no_grad():
      for ind, data in enumerate(non_memebr_loader, 0):
          inputs, labels = data
          inputs, labels = inputs.to(device), labels.to(device)
          queries = target_net(inputs)

          queries = torch.nn.functional.softmax(queries, dim=1)

          outputs = attack_model(queries)
          non_member_cali_list.extend(outputs.cpu().numpy())

    print("length of non_member_cali_list: " + str(len(non_member_cali_list)))
    logging.info(f"length of non_member_cali_list: {len(non_member_cali_list)}")
    print("average of non_member_cali_list : " + str(sum(non_member_cali_list)/len(non_member_cali_list)))
    logging.info(f"average of non_member_cali_list : {sum(non_member_cali_list)/len(non_member_cali_list)}")

    non_member_scores = []
    member_scores = []

    with torch.no_grad():
      for ind, data in enumerate(test_non_member_loader, 0):
          inputs, labels = data
          inputs, labels = inputs.to(device), labels.to(device)
          queries = target_net(inputs)

          queries = torch.nn.functional.softmax(queries, dim=1)

          scores = attack_model(queries)
          non_member_scores.extend(scores.cpu().numpy())

    with torch.no_grad():
      for ind, data in enumerate(test_member_loader, 0):
          inputs, labels = data
          inputs, labels = inputs.to(device), labels.to(device)
          queries = target_net(inputs)

          queries = torch.nn.functional.softmax(queries, dim=1)

          scores = attack_model(queries)
          member_scores.extend(scores.cpu().numpy())

    print("length of non_member_scores: " + str(len(non_member_scores)))
    print("length of member_scores: " + str(len(member_scores)))
    print("average non_member_scores : " + str(sum(non_member_scores)/len(non_member_scores)))
    print("average member_scores : " + str(sum(member_scores)/len(member_scores)))

    logging.info(f"length of non_member_scores: {len(non_member_scores)}")
    logging.info(f"length of member_scores: {len(member_scores)}")
    logging.info(f"average non_member_scores : {sum(non_member_scores)/len(non_member_scores)}")
    logging.info(f"average member_scores : {sum(member_scores)/len(member_scores)}")

    n_non_member = len(non_member_scores)
    scores = non_member_scores + member_scores
    p_values = []
    for i in range(len(scores)):
      p = (1 + len([j for j in non_member_cali_list if j <= scores[i]])) / (1 + len(non_member_cali_list))
      p_values.append(p)

    print("average p value : " + str(sum(p_values)/len(p_values)))
    logging.info("average p value : " + str(sum(p_values)/len(p_values)))

    return attack_model,n_non_member, p_values


def main():

    args = get_parameter()

    """# Create a log file"""
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)

    log_path = args.craftproj + '-main-log-%s' % (datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S"))
    log_path = log_path + '.txt'
    logging.basicConfig(
        filename=os.path.join(args.logdir, log_path),
        format="%(asctime)s - %(name)s - %(message)s",
        datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO, filemode='w')

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logging.info(str(args))

    if torch.cuda.is_available():
        # args.device = torch.device("cuda")
        cudnn.benchmark = True
    else:
        args.device = "cpu"
    print(f'device: {args.device}')
    logging.info(f'device: {args.device}')

    device = args.device
    learning_rate = args.lr
    n_target_epochs = args.epoch
    n_shadow = args.num_shadow
    n_shadow_epochs = args.epoch
    n_attack_epochs = args.epoch
    attack_learning_rate = args.attack_lr
    attack_hidden_layer_list = [20, 10]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])

    train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=None)
    test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    balenced_accuracy_classifier = []
    Precision_classifier = []
    auc_classifier = []
    TPR_classifier = []
    FPR_classifier = []
    # FDR_classifier = []
    Thres_classifier = []

    TPR_conform = []
    FPR_conform = []
    FDR_conform = []
    balenced_accuracy_conform = []
    Precision_list_conform = []
    AUC_conform_list = []

    start_time = time.time()
    for i in range(args.num_iters):
        print("==> iter:  " + str(i))
        # split datasets
        pri_tra_loader, pri_test_loader, non_memebr_cali_dataset, shadow_tra_loader_list, shadow_test_loader_list, test_member_loader, test_non_member_loader = data_random_split(args, logging, train_data, test_data)

        # train target models
        target_net = train_target_model(logging, pri_tra_loader, learning_rate, n_target_epochs, device, test_member_loader, test_non_member_loader)

        # train shadow models
        in_pred_list, out_pred_list = train_shadow_model(logging, shadow_tra_loader_list, shadow_test_loader_list, learning_rate, n_shadow_epochs, device, n_shadow=n_shadow)

        # train attack model
        attack_model, n_non_member, p_values = train_attack_classifier(args,logging, attack_hidden_layer_list, in_pred_list, out_pred_list, n_attack_epochs, non_memebr_cali_dataset, target_net, test_member_loader, test_non_member_loader, device, attack_learning_rate)

        # evaluate baseline
        balenced_accuracy, Precision, auc, TPR_list, FPR_list, Thres_list = evaluate_classifier(logging, target_net, attack_model, test_non_member_loader, test_member_loader, device)
        balenced_accuracy_classifier.append(balenced_accuracy)
        Precision_classifier.append(Precision)
        auc_classifier.append(auc)
        TPR_classifier.append(TPR_list)
        FPR_classifier.append(FPR_list)
        Thres_classifier.append(Thres_list)

        logging.info("^" * 80)
        logging.info("Balanced_accuracy_classifier for time: " + str(i) + " " + str(balenced_accuracy))
        logging.info("^" * 80)
        logging.info("Precision_classifier for time: " + str(i) + " " + str(Precision))
        logging.info("^" * 80)
        logging.info("AUC_classifier for time: " + str(i) + " " + str(auc))
        logging.info("^" * 80)
        logging.info("TPR_classifier for time: " + str(i) + " " + str(TPR_list))
        logging.info("^" * 80)
        logging.info("FPR_classifier for time: " + str(i) + " " + str(FPR_list))
        logging.info("^" * 80)
        logging.info("Thres_list for time: " + str(i) + " " + str(Thres_list))
        logging.info("^" * 80)

        # evaluate ours
        alphas_v = np.linspace(0.0, args.alpha, num=50)
        TPR_list, FPR_list, FDR_list, balenced_accuracy_list, auc_ours, Precision_list = evaluate_ours(args, logging, alphas_v, n_non_member, p_values)

        TPR_conform.append(TPR_list)
        FPR_conform.append(FPR_list)
        FDR_conform.append(FDR_list)
        balenced_accuracy_conform.append(balenced_accuracy_list)
        Precision_list_conform.append(Precision_list)
        AUC_conform_list.append(auc_ours)

        logging.info("~" * 80)
        logging.info("TPR_ours for time: " + str(i) + " " + str(TPR_list))
        logging.info("~" * 80)
        logging.info("FPR_ours for time: " + str(i) + " " + str(FPR_list))
        logging.info("~" * 80)
        logging.info("FDR_ours for time: " + str(i) + " " + str(FDR_list))
        logging.info("~" * 80)
        logging.info("Balanced_accuracy_ours for time : " + str(i) + " " + str(balenced_accuracy_list))
        logging.info("~" * 80)
        logging.info("Precision_list_ours for time : " + str(i) + " " + str(Precision_list))
        logging.info("~" * 80)
        logging.info("AUC_ours for time : " + str(i) + " " + str(auc_ours))
        logging.info("~" * 80)

        end_time = time.time()
        duration = end_time - start_time
        logging.info(f"Execution time : {duration} seconds")



if __name__ == '__main__':
    main()




    