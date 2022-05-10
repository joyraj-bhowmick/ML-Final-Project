import torch
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
from collections import namedtuple


def accuracy(y : np.ndarray, y_hat : np.ndarray) -> np.float64:
    """Calculate the simple accuracy given two numpy vectors, each with int values
    corresponding to each class.

    Args:
        y (np.ndarray): actual value
        y_hat (np.ndarray): predicted value

    Returns:
        np.float64: accuracy
    """
    ### TODO Implement accuracy function
    correct = 0
    for i in range(y.shape[0]):
        if (y[i] == y_hat[i]):
            correct = correct + 1
    return (correct/y.shape[0])
    
    #raise NotImplementedError


def approx_train_acc_and_loss(model, train_data : np.ndarray, train_labels : np.ndarray, weights) -> np.float64:
    """Given a model, training data and its associated labels, calculate the simple accuracy when the 
    model is applied to the training dataset.
    This function is meant to be run during training to evaluate model training accuracy during training.

    Args:
        model (pytorch model): model class object.
        train_data (np.ndarray): training data
        train_labels (np.ndarray): training labels

    Returns:
        np.float64: simple accuracy
    """
    train_labels = np.array(train_labels)
    # idxs = np.random.choice(len(train_data), 4000, replace=False)
    idxs = np.random.choice(len(train_data), 30, replace=False)
    x = torch.from_numpy(train_data[idxs].astype(np.float32))
    y = torch.from_numpy(train_labels[idxs].astype(np.int))
    logits = model(x)
    loss = F.cross_entropy(logits, y, weight=weights)
    y_pred = torch.max(logits, 1)[1]
    return accuracy(train_labels[idxs], y_pred.numpy()), loss.item()


def dev_acc_and_loss(model, dev_data : np.ndarray, dev_labels : np.ndarray, weights) -> np.float64:
    """Given a model, a validation dataset and its associated labels, calcualte the simple accuracy when the
    model is applied to the validation dataset.
    This function is meant to be run during training to evaluate model validation accuracy.

    Args:
        model (pytorch model): model class obj
        dev_data (np.ndarray): validation data
        dev_labels (np.ndarray): validation labels

    Returns:
        np.float64: simple validation accuracy
    """
    dev_labels = np.array(dev_labels)
    import pdb
    # pdb.set_trace()
    x = torch.from_numpy(dev_data.astype(np.float32))
    y = torch.from_numpy(dev_labels.astype(np.int))
    logits = model(x)
    loss = F.cross_entropy(logits, y, weight=weights)
    y_pred = torch.max(logits, 1)[1]
    return accuracy(dev_labels, y_pred.numpy()), loss.item()


def r_precision(target, prediction, max_n_prediction=500):
    '''R-precision evaluation metric'''
    pred = prediction[:max_n_prediction]
    targetset = set(target)
    
    denominator = len(targetset)
    numerator = float(len(set(pred[:denominator]).intersection(targetset)))
    
    r_prec_val = numerator/denominator
    return r_prec_val


def ndcg(relevant, retrieved, k, *args, **kwargs):
    '''Normalized Discounted Cumulative Gain'''
    dcg_val = dcg(relevant, retrieved, k)
    idcg_val = idcg(relevant, retrieved, k)
    
    if idcg_val == 0:
        raise ValueError("relevent is empty, divide by 0 error")
    
    ndcg_val = dcg_val / idcg_val
    return ndcg_val


def dcg(relevant, retrieved, k, *args, **kwargs):
    '''Discounted Cumulative Gain'''
    list1 = retrieved[:k]
    retrieved = list(OrderedDict.fromkeys(list1))
    relevant = list(OrderedDict.fromkeys(relevant))
    
    if (len(relevant) == 0 or len(retrieved) == 0):
        return 0.0
    
    else:
        rel_i = [float(el in relevant) for el in retrieved]
        rel_i_len = len(rel_i)+1
        
        i_variable = 1 + np.arange(1, rel_i_len)
        denominator = np.log2(i_variable)
        
        dcg_val = np.sum(rel_i/denominator)
        return dcg_val

def idcg(relevant, retrieved, k, *args, **kwargs):
    '''Ideal Discounted Cumulative Gain'''
    k_min = min(k, len(relevant))
    idcg_val = dcg(relevant, relevant, k_min)
    return idcg_val


Metrics = namedtuple('Metrics', ['r_precision', 'ndcg'])

def get_all_metrics(model, dev_data : np.ndarray, dev_labels : np.ndarray, weights):
    '''Return tuple of each evaluation metric'''
    dev_labels = np.array(dev_labels)
    x = torch.from_numpy(dev_data.astype(np.float32))
    y = torch.from_numpy(dev_labels.astype(np.int))
    logits = model(x)
    loss = F.cross_entropy(logits, y, weight=weights)
    y_pred = torch.max(logits, 1)[1]

    target = dev_labels
    prediction = y_pred
    k = len(y_pred)

    r_prec_val = r_precision(target, prediction, k)
    ndcg_val = ndcg(target, prediction, k)
    
    Metrics_val = Metrics(r_prec_val, ndcg_val)
    return Metrics_val

