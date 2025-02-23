"""
Evaluating the model on the test set.
"""

import torch
import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
'''
def evaluate(args,test_loader, net, criterion, device):  
    """
    Args:
        test_loader: an iterator over the test set.
        net: the neural network model employed.
        criterion: the loss function.
        device: denoting using CPU or GPU.

    Outputs:
        Average loss and accuracy achieved by the model in the test set.
    """    
    net.eval()

    accurate = 0
    loss = 0.0
    total = 0
    with torch.no_grad():
        for data in tqdm.tqdm(test_loader):
        #for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            
            #testnan=torch.isnan(outputs)
            #for x in testnan:
                #for y in x:
                    #if y:
                        #raise ValueError("output is nan")
                    
            loss += criterion(outputs, labels) * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            accurate += (predicted == labels).sum().item()
            total += labels.size(0)

        accuracy = 1.0 * accurate / total
        loss = loss.item() / total
        if(total==0):
            print("eroorrrrrr")

    return (loss, accuracy)
   '''

def evaluate(args, test_loader, net, criterion, device):
    """
    Args:
        test_loader: an iterator over the test set.
        net: the neural network model employed.
        criterion: the loss function.
        device: denoting using CPU or GPU.

    Outputs:
        Average loss, accuracy, F1-score, precision, and recall achieved by the model in the test set.
    """    
    net.eval()

    accurate = 0
    loss = 0.0
    total = 0
    all_predicted = []
    all_labels = []

    with torch.no_grad():
        for data in tqdm.tqdm(test_loader):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            
            loss += criterion(outputs, labels) * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            accurate += (predicted == labels).sum().item()
            total += labels.size(0)

            # Collect all predictions and labels for F1-score, precision, and recall calculation
            all_predicted.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        accuracy = 1.0 * accurate / total
        loss = loss.item() / total

        # Calculate F1-score, precision, and recall
        f1 = f1_score(all_labels, all_predicted, average='weighted')  # Use 'weighted' for multi-class classification
        precision = precision_score(all_labels, all_predicted, average='weighted')
        recall = recall_score(all_labels, all_predicted, average='weighted')
        
        conf_matrix = confusion_matrix(all_labels, all_predicted)

        if total == 0:
            print("error: no samples processed")

    return (loss, accuracy, f1, precision, recall, conf_matrix)
