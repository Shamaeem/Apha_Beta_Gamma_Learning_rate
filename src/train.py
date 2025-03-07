"""
Train a model on the training set.
"""
import torch
from load_optim import load_optim
from evaluate import evaluate
from torch.optim.lr_scheduler import ReduceLROnPlateau
import metrics
import time

def train(args, train_loader, test_loader, net, criterion, device):
    """
    Args:
        args: parsed command line arguments.
        train_loader: an iterator over the training set.
        test_loader: an iterator over the test set.
        net: the neural network model employed.
        criterion: the loss function.
        device: using CPU or GPU.

    Outputs:
        All training losses, training accuracies, test losses, and test
        accuracies on each evaluation during training.
    """
    '''print("args.optim_method")
    print(args.optim_method)
    print("Tmax:")
    print(args.train_epochs*len(train_loader))
    print(args.train_epochs)
    print(len(train_loader))'''
    optimizer = load_optim(params=net.parameters(),
                           optim_method=args.optim_method,
                           eta0=args.eta0,
                           alpha=args.alpha,
                           c=args.c,
                           milestones=args.milestones,
                           T_max=args.train_epochs*len(train_loader),
                           n_batches_per_epoch=len(train_loader),
                           nesterov=args.nesterov,
                           momentum=args.momentum,
                           weight_decay=args.weight_decay,
                           gamma=args.gamma,
                           coeff=args.coeff)

    if args.optim_method == 'SGD_ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer,
                                      mode='min',
                                      factor=args.alpha,
                                      patience=args.patience,
                                      threshold=args.threshold)

    # Choose loss and metric function
    loss_function = metrics.get_metric_function('softmax_loss')

    all_train_losses = []
    all_train_accuracies = []
    all_test_losses = []
    all_test_accuracies = []
    all_train_f1=[]
    all_train_precision=[]
    all_train_recalls=[]
    all_test_f1=[]
    all_test_precision=[]
    all_test_recalls=[]
    all_grad_norm=[]
    all_step_length=[]
    counter=-1
    Time=[]
    start_time=time.time()
    for epoch in range(1, args.train_epochs + 1):
        net.train()
        counter=counter+1
        for data in train_loader:
            inputs, labels = data
            if args.dataset == 'Flower102':
                labels = torch.sub(labels, 1)
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()        

            if args.optim_method.startswith('SLS'):
                closure = lambda : loss_function(net, inputs, labels, backwards=False)
                optimizer.step(closure)
                
            elif args.optim_method.startswith('SCGS'):
                closure = lambda net : loss_function(net, inputs, labels, backwards=False)
                l,g=optimizer.step(net,counter,closure) 
                all_grad_norm.append(g)
                
            elif args.optim_method.startswith('SCGWSA'):
                outputs = net(inputs)
                closure = lambda : criterion(outputs, labels)
                optimizer.step(net,counter,closure)  
            else:
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                if 'Polyak' in args.optim_method:
                    optimizer.step(loss.item())
                else:
                    optimizer.step()

        # Evaluate the model on training and validation dataset.
        if args.optim_method == 'SGD_ReduceLROnPlateau' or (epoch % args.eval_interval == 0):
            train_loss, train_accuracy, train_f1, train_precision, train_recall, train_conf_matix = evaluate(args,train_loader, net, criterion, device)
            
            end_epoch=time.time()
            Time.append(end_epoch-start_time)
            
            all_train_losses.append(train_loss)
            all_train_accuracies.append(train_accuracy)
            all_train_f1.append(train_f1)
            all_train_precision.append(train_precision)
            all_train_recalls.append(train_recall)

            test_loss, test_accuracy, test_f1, test_precision, test_recall, test_conf_matix = evaluate(args,test_loader, net, criterion, device)
            
            all_test_losses.append(test_loss)
            all_test_accuracies.append(test_accuracy)
            all_test_f1.append(test_f1)
            all_test_precision.append(test_precision)
            all_test_recalls.append(test_recall)

            print('Epoch %d --- ' % (epoch),
                  'train: loss - %g, ' % (train_loss),
                  'accuracy - %g; ' % (train_accuracy),
                  'f1 - %g; ' % (train_f1),
                  'precision - %g; ' % (train_precision),
                  'recall - %g; ' % (train_recall),
                  'test: loss - %g, ' % (test_loss),
                  'accuracy - %g' % (test_accuracy),
                  'f1 - %g' % (test_f1),
                  'precision - %g' % (test_precision),
                  'recall - %g' % (test_recall))

            if args.optim_method == 'SGD_ReduceLROnPlateau':
                scheduler.step(test_loss)

    return (all_train_losses, all_train_accuracies, 
    		all_test_losses, all_test_accuracies, Time, all_train_precision, all_train_recalls, all_train_f1, all_test_precision, all_test_recalls,all_test_f1)
            
