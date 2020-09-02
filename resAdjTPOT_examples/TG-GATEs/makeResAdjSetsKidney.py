import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import train_test_split
import os

inFile = sys.argv[1]
indexCol = sys.argv[2]
targetCol = sys.argv[3]
outDir = sys.argv[4]
type = sys.argv[5]

data = pd.read_csv(inFile, sep='\t', index_col=indexCol)
Xdata = data.drop(targetCol, axis=1)
Ydata = data[targetCol]
del data

for seed in range(1, 101):
    print(str(seed))
    if (type == 'classification'):
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(Xdata, Ydata, random_state=42+seed, train_size=0.75, test_size=0.25, stratify=Ydata)
    else:
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(Xdata, Ydata, random_state=42+seed, train_size=0.75, test_size=0.25)
    Ytrain = pd.DataFrame(Ytrain, index=Xtrain.index)
    Ytest = pd.DataFrame(Ytest, index=Xtest.index)

    dataTrain = pd.concat([Xtrain, Ytrain], axis = 1)
    dataTest = pd.concat([Xtest, Ytest], axis = 1)
    dataTrain.to_csv(outDir + "/train_" + str(seed), sep='\t', header=True, index_label=indexCol)
    dataTest.to_csv(outDir + "/test_" + str(seed), sep='\t', header=True, index_label=indexCol)

    os.system("python tpot/builtins/resAdjTpotPreprocessor.py " + outDir + "/train_" + str(seed) + " " + outDir + "/test_" + str(seed) + " " + indexCol + " " + targetCol + " " + type + " 5 " + outDir +  "/trainAdj_" + str(seed) + " " + outDir + "/testAdj_" + str(seed) + " DoseOrd,SacriOrd,COMPOUND_1,COMPOUND_2,COMPOUND_3,COMPOUND_4,COMPOUND_5,COMPOUND_6")
    os.system("rm -f " + outDir + "/train_" + str(seed) + " " + outDir + "/test_" + str(seed))

