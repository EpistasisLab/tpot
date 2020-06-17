"""
AUTHOR
Elisabetta Manduchi

SCOPE
Pre-processing script for resAdjTPOT

DESCRIPTION
This script prepares the training input for resAdjTPOT. The latter equals the
input file plus N additional column pairs (N=number of CV splits), where
each pair corresponds to an indicator column for training and testing
(denoting  training and testing rows for that split) and a column holding the
adjusted response variable (via residuals). The adjustment uses a linear or
logistic regression fitted on the training subset only. In addition, the
original target variable y is replaced by its adjstusment using the entire 
training set. Moreover, columns no longer needed for future usage can be removed from the final output.
The script also takes an optional hold-out testing file and preprocesses it
so that it can be scored by resAdjTPOT fitted pipeline. If the testing file
is provided, it is assumed that it has the same names for the index, target,
and covariate columns as the training input file.

ARGUMENTS
1. Input training file.
2. Optional input testing file (file path or None).
3. Column name for the index column (sample ids) if present.
4. Column name for the response variable y.
5. One of 'regression' or 'classification', depending on y.
6. Number of CV splits to generate.
7. Output training file.
8. Output testing file (file path or None).
9. Comma-separated list (no spaces) of the columns to use to adjust y.
10. Optional comma-separated list (no spaces) of columns to remove from
the output, as not needed for TPOT run.

EXAMPLE USAGE
python resAdjTpotPreprocessor.py inputTrain.txt None ids class classification 5 outputTrain.txt None age,sex,array age,sex
"""

import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split

if len(sys.argv)<10:
        sys.exit("Insufficient number of arguments passed in")

inTrainFile = sys.argv[1]
inTestFile = sys.argv[2]
indexCol = sys.argv[3]
targetCol = sys.argv[4]
mode = sys.argv[5]
numSplits = int(sys.argv[6])
outTrainFile = sys.argv[7]
outTestFile = sys.argv[8]
cov = sys.argv[9].split(',')

if (inTestFile is None and outTestFile is not None) or (inTestFile is not None and outTestFile is None):
        raise ValueError("If one of the names for input Test file or output Test file is provided, the other has to be provided too")

if (mode!='regression' and mode!='classification'):
        sys.exit("mode must be one of 'regression' or 'classification'")

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

data = pd.read_csv(inTrainFile, sep='\t', index_col=indexCol)
Xdata = data.drop(targetCol, axis=1)
Ydata = data[targetCol]
B = data[cov]

if inTestFile != 'None':
        testData = pd.read_csv(inTestFile, sep='\t', index_col=indexCol)
        Ydata_test = testData[targetCol]
        B_test = testData[cov]

for i in range(numSplits+1):
        if mode == 'classification':
                estimator = LogisticRegression(penalty='none',
                                               solver='lbfgs',
                                               multi_class='auto',
                                               max_iter=500)
        else:
                estimator = LinearRegression()

        if i==0:
                estimator.fit(B, Ydata)
                Yadj = pd.Series(index=data.index, dtype=float)
                Yadj = Ydata - getPredictions(estimator, B)
                Yadj.rename('adjY', inplace=True)
                data = data.join(Yadj)

                if inTestFile != 'None':
                        Yadj_test = pd.Series(index=testData.index, dtype=float)
                        Yadj_test = Ydata_test - getPredictions(estimator, B_test)
                        Yadj_test.rename('adjY', inplace=True)
                        testData = testData.join(Yadj_test)
                        oneData = np.ones(len(Ydata_test), dtype=int)
                        indicator1 = pd.Series(oneData, index=testData.index, name='indicator_1')
                        adjY1 = Yadj_test.copy()
                        adjY1.rename('adjY_1', inplace=True)
                        testData = testData.join([indicator1, adjY1])  # Mock placeholder columns needed for the pipeline to run on the testing set
        else:
                seed = 42 + i
                if mode == 'regression':
                        Xtrain, Xtest, Ytrain, Ytest = train_test_split(Xdata, Ydata, random_state=int(seed), train_size=0.75, test_size=0.25)
                else:
                        Xtrain, Xtest, Ytrain, Ytest = train_test_split(Xdata, Ydata, random_state=int(seed), train_size=0.75, test_size=0.25, stratify=Ydata)


                zeroData = np.zeros(len(Ydata), dtype=int)
                indicator = pd.Series(zeroData, index=data.index, name='indicator_' + str(i))
                indicator[Xtest.index] = 1

                Btrain = B.loc[Xtrain.index]
                Btest = B.loc[Xtest.index]

                estimator.fit(Btrain, Ytrain)
                Yadj = pd.Series(index=data.index, dtype=float, name='adjY_' + str(i))

                Yadj[Xtrain.index] = Ytrain - getPredictions(estimator, Btrain)
                Yadj[Xtest.index] = Ytest - getPredictions(estimator, Btest)

                data = data.join(indicator)
                data = data.join(Yadj)

data.drop(targetCol, axis=1, inplace=True)
if inTestFile != 'None':
        testData.drop(targetCol, axis=1, inplace=True)

if len(sys.argv)>10:
        remove = sys.argv[10].split(',')
        data.drop(remove, axis=1, inplace=True)
        if inTestFile != 'None':
                testData.drop(remove, axis=1, inplace=True)

data.to_csv(outTrainFile, sep='\t', header = True)
if inTestFile != 'None':
        testData.to_csv(outTestFile, sep='\t', header=True)
