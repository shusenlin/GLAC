from utils_data import prepare_cv_datasets, make_cv_mpu_train_set, make_uci_mpu_train_set, gen_index_dataset
import torch
from utils_loss import gce_loss
import argparse
from torch.utils.data import DataLoader
from utils_model import linear_model, mlp_model
from algorithms import NRPR
from utils_func import KernelPriorEstimator
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-lr', type=float, default=1e-2)
parser.add_argument('-wd', type=float, default=1e-4)
parser.add_argument('-ds', type=str, help='specify a dataset', default='har')
parser.add_argument('-uci', type=int, help='UCI dataset or not', default=1, choices=[1,0])
parser.add_argument('-ep', type=int, default=1500)
parser.add_argument('-seed', type=int, default=1)
parser.add_argument('-gpu', type=str, default='0')
parser.add_argument('-mo', type=str, help='specify a model', default='linear', choices=['mlp','linear'])
parser.add_argument('-iter', type=int, default=50)
parser.add_argument('-bs', type=int, default=500)
parser.add_argument('-lamda', type=float, default=1.2)
parser.add_argument('-q', type=float, default=0.1)
parser.add_argument('-t', type=int, default=1)
args = parser.parse_args()

device = torch.device("cuda:"+args.gpu if torch.cuda.is_available() else "cpu")
torch.manual_seed(args.seed);torch.cuda.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed),np.random.seed(args.seed);
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


loss_fn = gce_loss 

        
# number of labeled, unlabeled, and test per-class
if args.ds == 'mnist':
    num_labeled = 4000
    num_unlabeled = 1000
    num_test = 100      


if args.uci ==1:
    X_labeled, Y_labeled, X_unlabeled, Y_unlabeled, X_test, y_test = make_uci_mpu_train_set(args.ds, 500, 1000, 1000, args.seed)
else:
    full_train_loader, full_test_loader = prepare_cv_datasets(args.ds)
    X_labeled, Y_labeled, X_unlabeled, Y_unlabeled, X_test, y_test = make_cv_mpu_train_set(full_train_loader, full_test_loader, num_labeled, num_unlabeled, num_test, args)
	
# create the zero label matrix for the unlabeled data
[dim1, dim2] = Y_unlabeled.shape
pseudo_Y_unlabeled_train = torch.zeros(dim1, dim2) 

trainX = torch.cat((X_labeled, X_unlabeled), dim=0)
trainY = torch.cat((Y_labeled, pseudo_Y_unlabeled_train), dim=0)


trainset = gen_index_dataset(trainX, trainY)
label_train = gen_index_dataset(X_labeled, Y_labeled[:,:-1])

out_dim = Y_labeled.shape[1]

#choice model: linear for UCI dataset, mlp for image dataset
if args.mo == 'linear':
    model = linear_model(X_labeled.shape[1], out_dim).to(device)   
elif args.mo == 'mlp':
    model = mlp_model(784,500,out_dim).to(device)
    
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

if args.uci == 1: 
    train_dataloader = DataLoader(trainset, batch_size=trainX.shape[0], shuffle=True) 
    theta = KernelPriorEstimator(X_labeled, X_unlabeled, 5)
else: 
    train_dataloader = DataLoader(trainset, batch_size=args.bs, shuffle=True)
    theta = 0.8
acc, Macro_F1, AUC = NRPR(train_dataloader, X_test, y_test, model, optimizer, loss_fn, theta, device, args)

print(args)
print('Mean acc of last 10 epoch:', acc)
print('Mean Macro_F1 of last 10 epoch:', Macro_F1)
print('Mean AUC of last 10 epoch:', AUC)