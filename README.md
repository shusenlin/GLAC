# GLAC
This is the code for the paper: A Generalized Unbiased Risk Estimator for Learning with Augmented Classes (AAAI 2023).

The environment is as bellow:  
- Python 3.7
- numpy 1.21.5
- scipy 1.7.3
- scikit-learn 1.0.2
- pytorch 1.12.0
- torchvision 0.13.0 



To directly run this code, please refer to the following examples:

python main.py -ds har -mo linear -uci 1 -ep 1500 -lo gce -q 0.1 -t 1 -lamda 1.2 -lr 1e-2 -wd 1e-4 -gpu 0 -seed 1
python main.py -ds mnist -mo mlp -uci 0 -ep 200 -lo gce -q 0.7 -t 3 -lamda 1.0 -lr 1e-4 -wd 1e-3 -gpu 0 -seed 1


Thank you for your time!
