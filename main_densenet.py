from lib2to3.pgen2.literals import test
import sys
import csv
import os
import numpy as np
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.io_argparse import get_args
from utils.accuracies import (dev_acc_and_loss, accuracy, approx_train_acc_and_loss, get_all_metrics)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class TwoLayerDenseNet(torch.nn.Module):
    def __init__(self, input_shape, hidden_layer_width, n_classes):
        super().__init__()
        self.inputShape = input_shape
        self.hiddenLayerWidth = hidden_layer_width
        self.nClasses = n_classes

        self.linear1 = torch.nn.Linear(self.inputShape, self.hiddenLayerWidth)
        self.dropout = torch.nn.Dropout(p = 0.1)
        self.linear2 = torch.nn.Linear(self.hiddenLayerWidth, 24)
        self.linear3 = torch.nn.Linear(24, 48)
        self.linear4 = torch.nn.Linear(48, 96)
        self.linear5 = torch.nn.Linear(96, self.nClasses)
    
        # self.layer1 = torch.nn.Linear(self.inputShape, self.hiddenLayerWidth)
        # self.layer2 = torch.nn.Linear(self.hiddenLayerWidth, self.hiddenLayerWidth*2)
        # self.layer3 = torch.nn.Linear(self.hiddenLayerWidth, self.nClasses)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear3(x)
        x = F.relu(x)
        x = self.linear4(x)
        x = F.tanh(x)
        x = self.linear5(x)
        x = F.softmax(x)

        # x = self.layer1(x)
        # x = F.leaky_relu(x)
        # # x = self.layer2(x)
        # # x = F.leaky_relu(x)
        # x = self.layer3(x)
        # x = F.softmax(x)
        return (x)

if __name__ == "__main__":
    arguments = get_args(sys.argv)
    MODE = arguments.get('mode')
    DATA_DIR = arguments.get('data_dir')
    torch.set_printoptions(precision=8)
    
    
    if MODE == "train":
        
        LOG_DIR = arguments.get('log_dir')
        MODEL_SAVE_DIR = arguments.get('model_save_dir')
        LEARNING_RATE = arguments.get('lr')
        BATCH_SIZE = arguments.get('bs')
        EPOCHS = arguments.get('epochs')
        DATE_PREFIX = datetime.datetime.now().strftime('%Y%m%d%H%M')
        if LEARNING_RATE is None: raise TypeError("Learning rate has to be provided for train mode")
        if BATCH_SIZE is None: raise TypeError("batch size has to be provided for train mode")
        if EPOCHS is None: raise TypeError("number of epochs has to be provided for train mode")
        
        # Training data
        TRAIN_DATA = np.load("datasets/traindata.npy")
        TRAIN_LABELS = np.load("datasets/trainlabels.npy")

        import pdb
        # pdb.set_trace()
        counts = np.array([np.count_nonzero(np.array(TRAIN_LABELS)==x) for x in set(TRAIN_LABELS)])
        weights = sum(counts)/(100*counts)
        weights_torch = torch.from_numpy(weights).float()
        TRAIN_DATA, DEV_DATA, TRAIN_LABELS, DEV_LABELS = train_test_split(TRAIN_DATA, list(TRAIN_LABELS), test_size=0.05, random_state=2)

        # Constants
        (N_DATA, N_FEATURES) = TRAIN_DATA.shape
        N_CLASSES = 100
        (N_DEV_DATA, N_DEV_FEATURES) = DEV_DATA.shape
        # Normalize data
        train_data = TRAIN_DATA.copy()
        train_data = (train_data-train_data.mean(axis=1).reshape(N_DATA,1))/train_data.std(axis=1).reshape(N_DATA,1)
        dev_data = DEV_DATA.copy()
        dev_data = (dev_data-dev_data.mean(axis=1).reshape(N_DEV_DATA,1))/dev_data.std(axis=1).reshape(N_DEV_DATA,1)


        # do not touch the following 4 lines (these write logging model performance to an output file 
        # stored in LOG_DIR with the prefix being the time the model was trained.)
        LOGFILE = open(os.path.join(LOG_DIR, f"densenet.log"),'w')
        log_fieldnames = ['step', 'train_loss', 'train_acc', 'dev_loss', 'dev_acc']
        logger = csv.DictWriter(LOGFILE, log_fieldnames)
        logger.writeheader()
        
        model = TwoLayerDenseNet(N_FEATURES, 100, N_CLASSES)
                
        optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

        #print(train_data.shape, dev_data.shape)
        #print(TRAIN_LABELS.shape, DEV_LABELS.shape)

        step_list = []
        train_loss_list = []
        dev_acc_list = []
        r_list = []
        ndcg_list = []
        
        for step in range(EPOCHS):
            # pdb.set_trace()
            i = np.random.choice(train_data.shape[0], size=BATCH_SIZE, replace=False)
            x = torch.from_numpy(train_data[i].astype(np.float32))
            y = torch.from_numpy(np.array(TRAIN_LABELS)[i].astype(int))

             # Forward pass: Get logits for x
            logits = model(x)
            # Compute loss
            loss = F.cross_entropy(logits, y, weight=weights_torch)
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            
            if step % 100 == 0:
                # print(logits, logits.size())
                import pdb
                # pdb.set_trace()
                train_acc, train_loss = approx_train_acc_and_loss(model, train_data, TRAIN_LABELS, weights_torch)
                dev_acc, dev_loss = dev_acc_and_loss(model, dev_data, DEV_LABELS, weights_torch)
                r_squared, ndcg = get_all_metrics(model, dev_data, DEV_LABELS, weights_torch)

                step_list.append(step)
                train_loss_list.append(train_loss)
                dev_acc_list.append(dev_acc)
                r_list.append(r_squared)
                ndcg_list.append(r_list)

                step_metrics = {
                    'step': step, 
                    'train_loss': loss.item(), 
                    'train_acc': train_acc,
                    'dev_loss': dev_loss,
                    'dev_acc': dev_acc
                }

                print(f"On step {step}:\tTrain loss {train_loss}\t|\tDev acc is {dev_acc}")
                logger.writerow(step_metrics)
        LOGFILE.close()
        
        model_savepath = os.path.join(MODEL_SAVE_DIR,f"densenet.pt")

        plt.plot(step_list, r_list)
        plt.xlabel('Steps')
        plt.ylabel('R Squared')
        plt.savefig("r_squared")
        print("r_squared: {0}".format(r_list))

        plt.plot(step_list, ndcg_list)
        plt.xlabel('Steps')
        plt.ylabel('NDGC')
        plt.savefig("ndcg")
        print("ndcg_list: {0}".format(ndcg_list))

        plt.plot(step_list, train_loss_list)
        plt.xlabel('Steps')
        plt.ylabel('Training Loss')
        plt.savefig("training_loss")

        plt.plot(step_list, dev_acc_list)
        plt.xlabel('Steps')
        plt.ylabel('Dev Accuracy')
        plt.savefig("dev_accuracy")

        print("Training completed, saving model at {0}".format(model_savepath))
        torch.save(model, model_savepath)
        
        
    elif MODE == "predict":
        PREDICTIONS_FILE = arguments.get('predictions_file')
        WEIGHTS_FILE = arguments.get('weights')
        if WEIGHTS_FILE is None : raise TypeError("for inference, model weights must be specified")
        if PREDICTIONS_FILE is None : raise TypeError("for inference, a predictions file must be specified for output.")
        # Testing data
        TEST_DATA = np.load("datasets/testdata.npy")
        
        model = torch.load(WEIGHTS_FILE)
        
        predictions = []
        
        (N_TEST_DATA, N_TEST_FEATURES) = TEST_DATA.shape
        test_data = TEST_DATA.copy()
        test_data = (test_data-test_data.mean(axis=1).reshape(N_TEST_DATA,1))/test_data.std(axis=1).reshape(N_TEST_DATA,1)
        
        for test_case in test_data:       
            x = torch.from_numpy(test_case.astype(np.float32))
            #x = x.view(1,-1)
            logits = model(x)
            logits = torch.unsqueeze(logits, dim=0)
            #print(logits, logits.size())
            pred = torch.max(logits, 1)[1]
            predictions.append(pred.item())
        print(f"Storing predictions in {PREDICTIONS_FILE}")
        predictions = np.array(predictions)
        np.savetxt(PREDICTIONS_FILE, predictions, fmt="%d")
        
    else: raise Exception("Mode not recognized")