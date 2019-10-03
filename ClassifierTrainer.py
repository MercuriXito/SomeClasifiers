#-*-coding:utf-8-*-

"""
    @file:			Classifier_trainer.py
    @autor:			Victor Chen
    @description:
        简单的分类器训练类
"""

import torch 
import torch.nn as nn 
import torch.functional as F
import torch.optim as optim

import numpy as np 
import matplotlib.pyplot as plt

import time
from copy import deepcopy

models_root = "models/"

class ClassfierTrainer(object):
    """最简单的训练一个分类器的方法，并且保存在验证集上表现最好的模型
    """
    def __init__(self, lr = 10e-3, epochs = 20):
        # parameter initialization
        self.lr = lr
        self.epochs = epochs
        pass 

    def train(self, classifier, trainloader, valloader, device = None, \
        optimizer = None, criterion = None, lr_scheduler = None):
        """ Train classifier on trainloader and validate it on valloader. Then 
        select the best model and save it.
        """
        # constant
        train_batch_size = trainloader.batch_size 
        val_batch_size = valloader.batch_size

        interval = len(trainloader) // 10
        since = time.clock()

        # best 
        best_val_accuracy = 0.0
        best_state_dict = classifier.state_dict()

        # device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print("Train using device:{}".format(device))
        classifier.to(device)

        # methods
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        if optimizer is None:
            optimizer = optim.Adam(classifier.parameters(), self.lr)
        if lr_scheduler is None:
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 2, 0.3)

        for epoch in range(self.epochs):

            print("Epoch:{}/{}".format(epoch+1, self.epochs))
            correct_sum = 0
            total_sum = 0

            losses = []
            # train
            classifier.train()
            for i, data in enumerate(trainloader):
                images, labels = data
                images, labels = images.to(device), labels.to(device)

                predicted = classifier(images)
                loss = criterion(predicted, labels)
                # bp
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if i % interval == 0:
                    print("Iteration %7d: [ loss: %.12f ]" %(i + 1, loss.item()))
                    losses.append(loss.item())

            classifier.eval()
            for data in valloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)

                predicted = classifier(images)
                plabels = torch.argmax(predicted, dim=1)
                correct_sum += torch.sum(plabels == labels).item()
                total_sum += val_batch_size

            accuracy = correct_sum / total_sum
            print("Accuracy on validation set: %.6f%%" %(accuracy * 100))

            if accuracy > best_val_accuracy:
                best_val_accuracy = accuracy
                best_state_dict = deepcopy(classifier.state_dict())

            lr_scheduler.step()

        # save the best model
        torch.save(best_state_dict, models_root + classifier.__class__.__name__ + ".pth")

        elapse = time.clock() - since
        print("Training Using {:0f}min {:0f}s".format(elapse // 60, elapse % 60))
        print("Best validation accuracy:{:4f}%".format(best_val_accuracy * 100))

        return classifier.load_state_dict(best_state_dict)


    def test(self, classifier, dataloader,  device = None):
        """ return the accuracy of classifier performed on dataloder. 
        """
        assert(isinstance(dataloader, torch.utils.data.DataLoader))

        training = classifier.training
        batch_size = dataloader.batch_size

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Test using device:{}".format(device))
        classifier.to(device)

        correct_sum = 0
        total_sum = 0

        classifier.eval()
        for data in dataloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            predicted = classifier(images)
            plabels = torch.argmax(predicted, dim=1)
            correct_sum += torch.sum(plabels == labels).item()
            total_sum += batch_size
        
        accuracy = correct_sum / total_sum
        print("Total Accuracy: %.6f%%" %(accuracy * 100) )

        classifier.train(training)

        return accuracy

    def inference(self, classifier, batch, device = None):
        """ infer on a batch with classifier
        """
        assert(isinstance(batch, torch.Tensor))
        training = classifier.training

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            classifier.to(device)

        images = batch.to(device)
        classifier.eval()

        output = classifier(images)
        labels = torch.argmax(output, 1).detach().cpu().numpy()

        classifier.train(training)
        return labels


if __name__ == "__main__":
    pass