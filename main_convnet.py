import sys
import csv
import os
import numpy as np
import datetime
import torch
import torch.nn.functional as F
from utils.io_argparse import get_args
from utils.accuracies import (dev_acc_and_loss, accuracy, approx_train_acc_and_loss)


class TooSimpleConvNN(torch.nn.Module):
    def __init__(self, input_height, input_width, n_classes):
        super().__init__()
        
        
        ### TODO Implement Convnet architecture
        self.inputHeight = input_height
        self.inputWidth = input_width
        self.nClasses = n_classes

        self.conv1 = torch.nn.Conv2d(1, 8, 3)
        self.conv2 = torch.nn.Conv2d(8, 16, 3, 2)
        self.conv3 = torch.nn.Conv2d(16, self.nClasses, 1)
        
        #raise NotImplementedError
        
    def forward(self, x):
        
        ### TODO Implement feed forward function
        x = torch.unsqueeze(x,1)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.avg_pool2d(x, 12)
        x = self.conv3(x)

        x = torch.squeeze(x)
        return (x)
        
        #raise NotImplementedError
    
    
    
if __name__ == "__main__":
    arguments = get_args(sys.argv)
    MODE = arguments.get('mode')
    DATA_DIR = arguments.get('data_dir')
    
    
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
        TRAIN_IMAGES = np.load(os.path.join(DATA_DIR, "fruit_images.npy"))
        TRAIN_LABELS = np.load(os.path.join(DATA_DIR, "fruit_labels.npy"))
        DEV_IMAGES = np.load(os.path.join(DATA_DIR, "fruit_dev_images.npy"))
        DEV_LABELS = np.load(os.path.join(DATA_DIR, "fruit_dev_labels.npy"))
        
        ### TODO get the following parameters and name them accordingly: 
        # [N_IMAGES] Number of images in the training corpus 
        # [HEIGHT] Height and [WIDTH] width dimensions of each image
        # [N_CLASSES] number of output classes
        (N_IMAGES, WIDTH, HEIGHT) = TRAIN_IMAGES.shape
        N_CLASSES = 6
        N_DEV_IMGS = DEV_IMAGES.shape[0]
        FLATTEN_DIM = WIDTH*HEIGHT
        
        #raise NotImplementedError        
        
        
        ### TODO Normalize each of the individual images to a mean of 0 and a variance of 1
        flat_train_imgs = np.ndarray((N_IMAGES,WIDTH*HEIGHT))
        flat_dev_imgs = np.ndarray((N_DEV_IMGS,WIDTH*HEIGHT))

        flat_train_imgs = TRAIN_IMAGES.reshape((N_IMAGES,WIDTH*HEIGHT))
        flat_train_imgs = (flat_train_imgs-flat_train_imgs.mean(axis=1).reshape(N_IMAGES,1))/flat_train_imgs.std(axis=1).reshape(N_IMAGES,1)
        flat_train_imgs = flat_train_imgs.reshape((N_IMAGES,WIDTH,HEIGHT))

        flat_dev_imgs = DEV_IMAGES.reshape((N_DEV_IMGS,WIDTH*HEIGHT))
        flat_dev_imgs = (flat_dev_imgs-flat_dev_imgs.mean(axis=1).reshape(N_DEV_IMGS,1))/flat_dev_imgs.std(axis=1).reshape(N_DEV_IMGS,1)
        flat_dev_imgs = flat_dev_imgs.reshape((N_DEV_IMGS,WIDTH,HEIGHT))

        #raise NotImplementedError
        
        
        # do not touch the following 4 lines (these write logging model performance to an output file 
        # stored in LOG_DIR with the prefix being the time the model was trained.)
        LOGFILE = open(os.path.join(LOG_DIR, f"convnet.log"),'w')
        log_fieldnames = ['step', 'train_loss', 'train_acc', 'dev_loss', 'dev_acc']
        logger = csv.DictWriter(LOGFILE, log_fieldnames)
        logger.writeheader()
        
        # call on model
        model = TooSimpleConvNN(input_height = HEIGHT, input_width= WIDTH,
                                 n_classes=N_CLASSES)
        
        ### TODO (OPTIONAL) : you can change the choice of optimizer here if you wish.
        optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
        
        for step in range(EPOCHS):
            i = np.random.choice(flat_train_imgs.shape[0], size=BATCH_SIZE, replace=False)
            x = torch.from_numpy(flat_train_imgs[i].astype(np.float32))
            y = torch.from_numpy(TRAIN_LABELS[i].astype(np.int))
            
            
            # Forward pass: Get logits for x
            logits = model(x)
            # Compute loss
            loss = F.cross_entropy(logits, y)
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            
            if step % 100 == 0:
                train_acc, train_loss = approx_train_acc_and_loss(model, flat_train_imgs, TRAIN_LABELS)
                dev_acc, dev_loss = dev_acc_and_loss(model, flat_dev_imgs, DEV_LABELS)
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
        
        ### TODO (OPTIONAL) You can remove the date prefix if you don't want to save every model you train 
        ### i.e. "{DATE_PREFIX}_convnet.pt" > "convnet.pt"
        model_savepath = os.path.join(MODEL_SAVE_DIR,f"convnet.pt")
        
        print("Training completed, saving model at {model_savepath}")
        torch.save(model, model_savepath)
        
        
    elif MODE == "predict":
        PREDICTIONS_FILE = arguments.get('predictions_file')
        WEIGHTS_FILE = arguments.get('weights')
        if WEIGHTS_FILE is None : raise TypeError("for inference, model weights must be specified")
        if PREDICTIONS_FILE is None : raise TypeError("for inference, a predictions file must be specified for output.")
        TEST_IMAGES = np.load(os.path.join(DATA_DIR, "fruit_test_images.npy"))
        
        model = torch.load(WEIGHTS_FILE)
        
        predictions = []

        (N_TEST_IMGS,WIDTH,HEIGHT) = TEST_IMAGES.shape
        flat_test_imgs = np.ndarray((N_TEST_IMGS,WIDTH*HEIGHT))
        flat_test_imgs = TEST_IMAGES.reshape((N_TEST_IMGS,WIDTH*HEIGHT))
        flat_test_imgs = (flat_test_imgs-flat_test_imgs.mean(axis=1).reshape(N_TEST_IMGS,1))/flat_test_imgs.std(axis=1).reshape(N_TEST_IMGS,1)
        TEST_IMAGES = flat_test_imgs.reshape((N_TEST_IMGS,WIDTH,HEIGHT))

        for test_case in TEST_IMAGES:
            ### TODO Normalize your test dataset  (identical to how you did it with training images)
            
            
            #raise NotImplementedError
        
        
            x = torch.from_numpy(test_case.astype(np.float32))
            x = torch.unsqueeze(x,0)
            logits = model(x)
            logits = torch.unsqueeze(logits,0)
            pred = torch.max(logits, 1)[1]
            predictions.append(pred.item())
        print(f"Storing predictions in {PREDICTIONS_FILE}")
        predictions = np.array(predictions)
        np.savetxt(PREDICTIONS_FILE, predictions, fmt="%d")


    else: raise Exception("Mode not recognized")
