import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder

    
def NRPR(train_dataloader, X_test, y_test, model, optimizer, loss_fn, theta, device, args):

    acc_list = []
    f1_list = [] 
    auc_list = [] 
    for epoch in range(args.ep):
        loss = 0
        count = 1 
        for X,Y,index in train_dataloader:
            batchX = X.to(device)
            batchY = Y.to(device)
            label_sum_for_each_instance = batchY.sum(dim=1) # number of examples in the batch
            unlabeled_index = (label_sum_for_each_instance == 0) # find the index of unlabeled data
            labeled_index = (label_sum_for_each_instance != 0)
            

            optimizer.zero_grad()  
            pred = model(batchX)

            pred_labeled = pred[labeled_index,:]
            labeled_batchY = batchY[labeled_index,:]            
            pred_unlabeled = pred[unlabeled_index, :]
            unlabeled_batchY = batchY[unlabeled_index, :]
            unlabeledY = torch.zeros_like(unlabeled_batchY).to(device)
            unlabeledY[:,-1] = 1
            
            n, k = labeled_batchY.shape[0], labeled_batchY.shape[1]
            temp_loss = torch.zeros(n, k).to(device)
            for i in range(k):
                tempY = torch.zeros(n, k).to(device)
                tempY[:, i] = 1.0 # calculate the loss with respect to the i-th label
                temp_loss[:, i] = loss_fn(pred_labeled, tempY, args.q)      

            loss_1_pos = (theta*(temp_loss*labeled_batchY).sum(dim=1)).mean()
            loss_1_neg = theta*temp_loss[:,-1].mean()
            
            loss_2 = loss_fn(pred_unlabeled, unlabeledY, args.q).mean()
                
            ## 
            if loss_2 - loss_1_neg >= 0:
                train_loss = loss_1_pos + loss_2 - loss_1_neg
            else:
                train_loss = loss_1_pos + loss_2 - loss_1_neg + args.lamda*(loss_1_neg-loss_2)**args.t                 
          
            train_loss.backward()
            optimizer.step()
            loss+= train_loss
            count+=1
        acc, Macro_F1, AUC = evaluate(model, X_test, y_test, device)

        if (epoch+1) % args.iter == 0:
            print('epoch:', epoch+1, 'loss:', str(loss.data.item()/count)[:8], 'acc:', str(acc)[:8],'F1:', str(Macro_F1)[:8], 'AUC:', str(AUC)[:8])    
        if epoch >= (args.ep-10):
            acc_list.append(acc)
            f1_list.append(Macro_F1)
            auc_list.append(AUC)
    return np.mean(acc_list), np.mean(f1_list), np.mean(auc_list)


def evaluate(model, X_test, y_test, device):
    X_test, y_test = X_test.to(device), y_test.to(device)
    outputs = model(X_test)
    pred = outputs.argmax(dim=1)

    logit = F.softmax(outputs, dim=1)
    label, predict, prediction_scores = y_test.cpu().numpy(), pred.detach().cpu().numpy(), logit.detach().cpu().numpy()
    one_hot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    one_hot_encoder.fit(np.array(label).reshape(-1, 1))
    true_scores = one_hot_encoder.transform(np.array(label).reshape(-1, 1))
    assert prediction_scores.shape == true_scores.shape
    AUC = roc_auc_score(true_scores, prediction_scores, multi_class='ovo')
    Macro_F1 = f1_score(label, predict, average='macro')
    accuracy = (pred==y_test).sum().item()/pred.shape[0]
    return accuracy, Macro_F1, AUC


