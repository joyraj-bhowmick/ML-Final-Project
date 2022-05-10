from lib2to3.pgen2.literals import test
import sys
import csv
import os
import numpy as np
import datetime
import torch
import torch.nn.functional as F
from utils.io_argparse import get_args
from utils.accuracies import (dev_acc_and_loss, accuracy, approx_train_acc_and_loss)


class TwoLayerDenseNet(torch.nn.Module):
    def __init__(self, input_shape, hidden_layer_width, n_classes):
        super().__init__()
        self.inputShape = input_shape
        self.hiddenLayerWidth = hidden_layer_width
        self.nClasses = n_classes

        self.hidden = torch.nn.Linear(self.inputShape, self.hiddenLayerWidth)
        self.predict = torch.nn.Linear(self.hiddenLayerWidth, 1)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        
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
        # Validation data
        DEV_DATA = np.load("datasets/devdata.npy")
        DEV_LABELS = np.load("datasets/devlabels.npy")
        # Constants
        (N_DATA, N_FEATURES) = TRAIN_DATA.shape
        N_CLASSES = 100
        (N_DEV_DATA, N_DEV_FEATURES) = DEV_DATA.shape
        # Normalize data
        train_data = TRAIN_DATA.copy()
        train_data = (train_data-train_data.mean(axis=1).reshape(N_DATA,1))/train_data.std(axis=1).reshape(N_DATA,1)
        train_labels = TRAIN_LABELS.copy()
        train_labels = train_labels.reshape(N_DATA,1)
        dev_data = DEV_DATA.copy()
        dev_data = (dev_data-dev_data.mean(axis=1).reshape(N_DEV_DATA,1))/dev_data.std(axis=1).reshape(N_DEV_DATA,1)
        dev_labels = DEV_LABELS.copy()
        dev_labels = dev_labels.reshape(N_DEV_DATA,1)
        
        LOGFILE = open(os.path.join(LOG_DIR, f"densenet.log"),'w')
        log_fieldnames = ['step', 'train_loss']
        logger = csv.DictWriter(LOGFILE, log_fieldnames)
        logger.writeheader()
        
        model = TwoLayerDenseNet(N_FEATURES, 30, N_CLASSES)
                
        optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

        loss_func = torch.nn.MSELoss()
        
        for step in range(EPOCHS):
            i = np.random.choice(train_data.shape[0], size=BATCH_SIZE, replace=False)
            x = torch.from_numpy(train_data[i].astype(np.float32))
            y = torch.from_numpy(train_labels[i].astype(np.float32))
            
            # Forward pass: Get prediction
            prediction = model(x)
            # Compute loss
            loss = loss_func(prediction, y)
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if step % 100 == 0:
                #train_acc, train_loss = approx_train_acc_and_loss(model, train_data, TRAIN_LABELS)
                #dev_acc, dev_loss = dev_acc_and_loss(model, dev_data, DEV_LABELS)
                step_metrics = {
                    'step': step, 
                    'train_loss': loss.item(), 
                }

                print(f"On step {step}:\tTrain loss {loss.item()}")
                logger.writerow(step_metrics)
        LOGFILE.close()
        
        model_savepath = os.path.join(MODEL_SAVE_DIR,f"densenet.pt")
        
        print("Training completed, saving model at {model_savepath}")
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
            pred = model(x)
            predictions.append(pred.item())
        print(f"Storing predictions in {PREDICTIONS_FILE}")
        predictions = np.array(predictions)
        np.savetxt(PREDICTIONS_FILE, predictions, fmt="%f")
        
    else: raise Exception("Mode not recognized")
