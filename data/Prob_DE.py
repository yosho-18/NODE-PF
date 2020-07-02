import warnings

warnings.filterwarnings('ignore')

from Prob_DE_calc import GBM
#import kakuritubibunhouteisiki
import csv
import numpy as np

numICs = 1000

#x1range = [-2, 2]
#tSpan = np.arange(0, 2.5 + 0.1, 0.25)# np.arange(0, 125, 0.25)  # 0:0.02:1
tSpan = np.arange(0.0, 1, 1 / 11)



def make_csv(filename, X):
    with open(filename, 'w') as csv_file:
        fieldnames = ['precision_x']#, 'precision_x2']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        for i in range(len(X)):
            writer.writerow({'precision_x': X[i]})#, 'precision_x2': X[i, 1]})


filenamePrefix = 'GBM'
seed = 10
X_train = GBM(numICs, tSpan, seed, "x") # 0, 2.5, 0.25
filename_train = filenamePrefix + '_train_x.csv'
make_csv(filename_train, X_train)

seed = 10
X_train = GBM(numICs, tSpan, seed, "y")
filename_train = filenamePrefix + '_train_y.csv'
make_csv(filename_train, X_train)

seed = 1
X_train = GBM(numICs, tSpan, seed)
filename_train = filenamePrefix + "_E_recon_10" + '.csv'
make_csv(filename_train, X_train)

seed = 3
X_train = GBM(10 * numICs, tSpan, seed)
filename_train = filenamePrefix + "_E_eigfunc" + '.csv'
make_csv(filename_train, X_train)