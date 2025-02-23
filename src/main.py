if __name__ == "__main__":
    import torch
    import torch.nn as nn

    import numpy as np
    import os
    import random

    from load_args import load_args
    from data_loader import data_loader
    from mnist_cnn import MNISTConvNet
    from cifar_cnn import CIFARConvNet
    from flower_cnn import FLOWERConvNet
    from cifar10_resnet import resnet20
    from cifar100_densenet import densenet
    from train import train
    from evaluate import evaluate

    def main():
        args = load_args()
        optim_method=args.optim_method
        for dataset in ['FashionMNIST','CIFAR10']:
            args.dataset=dataset
            print(args.dataset)
                
 
                       
            if not os.path.exists(args.log_folder):
                os.makedirs(args.log_folder)        
            
            for run_number in [1]:
                    
                # Check the availability of GPU.
                use_cuda = args.use_cuda and torch.cuda.is_available()
                device = torch.device("cuda:0" if use_cuda else "cpu")

                # Set the ramdom seed for reproducibility.
                if args.reproducible:
                    torch.manual_seed(args.seed)
                    np.random.seed(args.seed)
                    random.seed(args.seed)
                    if device != torch.device("cpu"):
                        torch.backends.cudnn.deterministic = True
                        torch.backends.cudnn.benchmark = False

                # set seed
                # ---------------
                #seed = 42 + run_number
                seed = 42
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
        
                print ("seed")
                print(seed)
                #----------------------------train epoches----------------------
                args.B=0
                    
                if args.dataset == 'FashionMNIST':
                    weight_decay=0.0001 
                    train_epochs=50  
                    if optim_method=='SGD_tan1_Decay':
                        eta0=0.02 
                    elif optim_method=='SGD_tan1_Decay':
                        eta0=0.02
                    elif optim_method=='SGD_tan1_Decay':
                        eta0=0.01                                        
                    #eta0=0.05 
                    #args.B=0.35                         
                elif args.dataset =='CIFAR10':
                    weight_decay=0.0001  
                    train_epochs=164  
                    eta0=0.2 
                    #args.B=0.25
                elif args.dataset =='CIFAR100':
                    weight_decay=0.0005  
                    train_epochs=60 
                    eta0=0.15
                    #args.B=0.25     
                            
                if optim_method=='SGD_Cosine_Decay':
                        eta0=0.09
                         
                args.weight_decay=weight_decay
                args.train_epochs=train_epochs
                args.eta0=eta0
                
                
                print("args.eta0:", args.eta0)
                    
                # Load data, note we will also call the validation set as the test set.
                print('Loading data...')
                dataset = data_loader(dataset_name=args.dataset,
                              dataroot=args.dataroot,
                              batch_size=args.batchsize,
                              val_ratio=(args.val_ratio if args.validation else 0))
                train_loader = dataset[0]
                if args.validation:
                    test_loader = dataset[1]
                else:
                    test_loader = dataset[2]

                 # Define the model and the loss function.
                if args.dataset == 'CIFAR10':
                    net = resnet20()
                elif args.dataset == 'CIFAR100':
                    net = densenet(depth=100, growthRate=12, num_classes=100)
                    #net = CIFARConvNet()
                elif args.dataset in ['MNIST', 'FashionMNIST']:
                    net = MNISTConvNet()
                else:
                    raise ValueError("Unsupported dataset {0}.".format(args.dataset))    
                net.to(device)
                criterion = nn.CrossEntropyLoss()

                # Train and evaluate the model.
                print("Training...")
                running_stats = train(args, train_loader, test_loader, net,
                              criterion, device)
                all_train_losses, all_train_accuracies, = running_stats[:2]
                all_test_losses, all_test_accuracies, Time = running_stats[2:5]
                all_train_precision,all_train_recall,all_train_f1,all_test_precision,all_test_recall,all_test_f1 = running_stats[5:]

                print("Evaluating...")
                final_train_loss, final_train_accuracy, final_train_f1, final_train_precision, final_train_recall,final_train_confusion_matrix = evaluate(args,train_loader, net,
                                                          criterion, device)
                final_test_loss, final_test_accuracy , final_test_f1, final_test_precision, final_test_recall, final_test_confusion_matrix= evaluate(args,test_loader, net,
                                                        criterion, device)

                # Logging results.
                print('Writing the results.')
                if not os.path.exists(args.log_folder):
                    os.makedirs(args.log_folder)
                              
                log_name = (('%s_%s_' % (args.dataset, args.optim_method))
                         + ('Eta0_%g_' % (args.eta0))
                         + ('b_%g_' % (args.B))
                         + ('Run_%g_' % (run_number))
                         + ('WD_%g_' % (args.weight_decay))
                         + (('Mom_%g_' % (args.momentum))
                         if args.optim_method.startswith('SGD') else '')
                         + (('alpha_%g_' % (args.alpha))
                         if args.optim_method not in ['Adam', 'SGD'] else '')
                         + (('Milestones_%s_' % ('_'.join(args.milestones)))
                         if args.optim_method == 'SGD_Stage_Decay' else '')
                         + (('c_%g_' % (args.c))
                         if args.optim_method.startswith('SLS') else '')
                         + (('Patience_%d_Thres_%g_' % (args.patience, args.threshold))
                         if args.optim_method == 'SGD_ReduceLROnPlateau' else '')
                         + ('Epoch_%d_Batch_%d_' % (args.train_epochs, args.batchsize))
                         + ('%s' % ('Validation' if args.validation else 'Test'))
                         + '.txt')
                         
                log_name2 = (('%s_%s_' % (args.dataset, args.optim_method))
                         + ('Eta0_%g_' % (args.eta0))
                         + ('b_%g_' % (args.B))
                         + ('Run_%g_' % (run_number))
                         + ('WD_%g_' % (args.weight_decay))
                         + (('Mom_%g_' % (args.momentum))
                         if args.optim_method.startswith('SGD') else '')
                         + (('alpha_%g_' % (args.alpha))
                         if args.optim_method not in ['Adam', 'SGD'] else '')
                         + (('Milestones_%s_' % ('_'.join(args.milestones)))
                         if args.optim_method == 'SGD_Stage_Decay' else '')
                         + (('c_%g_' % (args.c))
                         if args.optim_method.startswith('SLS') else '')
                         + (('Patience_%d_Thres_%g_' % (args.patience, args.threshold))
                         if args.optim_method == 'SGD_ReduceLROnPlateau' else '')
                         + ('Epoch_%d_Batch_%d_' % (args.train_epochs, args.batchsize))
                         + ('%s' % ('Validation' if args.validation else 'Test'))
                         + '_Time.txt')     
                    
                mode = 'w' if args.validation else 'a'
                with open(args.log_folder + '/' + log_name, mode) as f:
                       f.write('Training running losses:\n')
                       f.write('{0}\n'.format(all_train_losses))
                       f.write('Training running accuracies:\n')
                       f.write('{0}\n'.format(all_train_accuracies))
                       f.write('Training running precision:\n')
                       f.write('{0}\n'.format(all_train_precision))
                       f.write('Training running recall:\n')
                       f.write('{0}\n'.format(all_train_recall))
                       f.write('Training running f1:\n')
                       f.write('{0}\n'.format(all_train_f1))
                       f.write('Final training loss is %g\n' % final_train_loss)
                       f.write('Final training accuracy is %g\n' % final_train_accuracy)
                       f.write('Final training precision is %g\n' % final_train_precision)
                       f.write('Final training recall is %g\n' % final_train_recall)
                       f.write('Final training f1 is %g\n' % final_train_f1)
                       
                       train_conf_matrix_str = np.array2string(final_train_confusion_matrix)
                       f.write('Final training confusion matrix is:\n')
                       f.write(train_conf_matrix_str + '\n')
                      

                       f.write('Test running losses:\n')
                       f.write('{0}\n'.format(all_test_losses))
                       f.write('Test running accuracies:\n')
                       f.write('{0}\n'.format(all_test_accuracies))  
                       f.write('Test running precision:\n')
                       f.write('{0}\n'.format(all_test_precision))
                       f.write('Test running recall:\n')
                       f.write('{0}\n'.format(all_test_recall))
                       f.write('Test running f1:\n')
                       f.write('{0}\n'.format(all_test_f1))             
                       f.write('Final test loss is %g\n' % final_test_loss)
                       f.write('Final test accuracy is %g\n' % final_test_accuracy) 
                       f.write('Final test precision is %g\n' % final_test_precision)
                       f.write('Final test recall is %g\n' % final_test_recall)
                       f.write('Final test f1 is %g\n' % final_test_f1)
                       
                       test_conf_matrix_str = np.array2string(final_test_confusion_matrix)
                       f.write('Final test confusion matrix is:\n')
                       f.write(test_conf_matrix_str + '\n')
                       
                '''if not os.path.exists(args.log_folder + '/Time'):
                       os.makedirs(args.log_folder + '/Time')
                    
                    mode = 'w' if args.validation else 'a'  
                    with open(args.log_folder + '/Time/' + log_name2, mode) as f:
                       f.write('Time:\n')
                       f.write('{0}\n'.format(Time))
                '''
                    
                                   
        print('Finished.')
        #if not os.path.exists(args.log_step_length_folder):
            #os.makedirs(args.log_sten_length_folder)
        #step_length_log_name = (('%s_%s_' % (args.dataset, args.optim_method))
                   #  + ('Eta0_%g_' % (args.eta0))
                    # + '.txt')
        #mode = 'w' if args.validation else 'a'
        #with open(args.log_step_length_folder + '/' + step_length_log_name, mode) as f:
            #f.write('step_length:\n')
           # f.write('{0}\n'.format(all_step_length))

    main()
