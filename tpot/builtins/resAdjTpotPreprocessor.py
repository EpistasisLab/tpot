"""
AUTHOR
Elisabetta Manduchi

DATE
April 3, 2020

SCOPE
Pre-processing script for resAdjTPOT

DESCRIPTION
This script outputs a file which consists of the input file plus N additional
column pairs (N=number of CV splits), where each pair corresponds to an
indicator column (to denote training and testing rows for that split) and
a column holding the adjusted response variable (via residuals).
The adjustment uses a linear or logistic regression fitted on the training
subset only.
Moreover, columns no longer needed for future usage can be removed from the
final output.

ARGUMENTS
1. Input data file.
2. Column name for the response variable y.
3. One of 'regression' or 'classification', depending on y.
4. Number of CV splits to generate.
5. Output data file.
6. Comma-separated list (no spaces) of the columns to use to adjust y.
7. Optional comma-separated list (no spaces) of columns to remove from the output, as not needed for TPOT run.

EXAMPLE USAGE
python resAdjTpotPreprocessor.py input.txt class classification 5 output.txt age,sex,array age,sex
"""

import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split

if len(sys.argv)<7:
        sys.exit("Insufficient number of arguments passed in")

inFile = sys.argv[1]
targetCol = sys.argv[2]
mode = sys.argv[3]
numSplits = int(sys.argv[4])
outFile = sys.argv[5]
cov = sys.argv[6].split(',')

def getPredictions(estimator, B):
        if mode == 'classification':
            predProba = estimator.predict_proba(B)
            pi = np.zeros((predProba.shape[0], ))
            for idx, gt in enumerate(estimator.classes_):
                gt = int(gt)
                pi = pi + gt*predProba[:, idx]
        else:
            pi = estimator.predict(B)
        return pi

if (mode!='regression' and mode!='classification'):
    sys.exit("mode must be one of 'regression' or 'classification'")

data = pd.read_csv(inFile, sep='\t', index_col=0)
Xdata = data.drop(targetCol, axis=1)
Ydata = data[targetCol]
B = data[cov]

for i in range(numSplits):
    seed = 42 + i
    if mode == 'regression':
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(Xdata, Ydata, random_state=int(seed), train_size=0.75, test_size=0.25)
    else:
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(Xdata, Ydata, random_state=int(seed), train_size=0.75, test_size=0.25, stratify=Ydata)

    zeroData = np.zeros(len(Ydata), dtype=int)
    indicator = pd.Series(zeroData, index=data.index, name='indicator_' + str(i+1))
    indicator[Xtest.index] = 1

    if mode == 'classification':
        estimator = LogisticRegression(penalty='none',
                                   solver='lbfgs',
                                   multi_class='auto',
                                   max_iter=500)        
    else:
        estimator = LinearRegression()

    Btrain = B.loc[Xtrain.index]
    Btest = B.loc[Xtest.index]

    estimator.fit(Btrain, Ytrain)
    Yadj = pd.Series(index=data.index, dtype=float, name='adjY_' + str(i+1))
    Yadj[Xtrain.index] = Ytrain - getPredictions(estimator, Btrain)
    Yadj[Xtest.index] = Ytest - getPredictions(estimator, Btest)

    data = data.join(indicator)
    data = data.join(Yadj)

if len(sys.argv)>=7:
        remove = sys.argv[7].split(',')
        data = data.drop(remove, axis=1)

data.to_csv(outFile, sep='\t', header = True)
