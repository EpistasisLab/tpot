import pandas as pd
import numpy as np
import sys
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from tpot.config import classifier_config_dict
import eli5
from eli5.sklearn import PermutationImportance

seed = sys.argv[1]
inFile = sys.argv[2]
indexCol = sys.argv[3]
targetCol = sys.argv[4]
outPath = sys.argv[5]
n_gen = int(sys.argv[6])
n_pop = int(sys.argv[7])

data = pd.read_csv(inFile, sep='\t', index_col=indexCol)
Xdata = data.drop([targetCol, 'DoseOrd', 'SacriOrd', 'COMPOUND_1', 'COMPOUND_2', 'COMPOUND_3', 'COMPOUND_4', 'COMPOUND_5', 'COMPOUND_6'], axis=1)
Ydata = data[targetCol]
del data

#print(Xdata)
#print(Ydata)
#sys.exit()

Xtrain, Xtest, Ytrain, Ytest = train_test_split(Xdata, Ydata, random_state=42+int(seed), train_size=0.75, test_size=0.25, stratify=Ydata)

classifier_config_dict['tpot.builtins.FeatureSetSelector'] = {
    'subset_list': ['fsProbesets2orMoreGenes.csv'],
    'sel_subset': range(154),
}

tpot = TPOTClassifier(generations=n_gen, population_size=n_pop, verbosity=2, 
                      random_state=42+int(seed), scoring="balanced_accuracy",
                      config_dict=classifier_config_dict, 
                      template='FeatureSetSelector-Transformer-Classifier')

tpot.fit(Xtrain, Ytrain)

tpot.export(outPath + '/pipelines/pipeline_' + seed + '.py')
scores = []
scores.append([tpot._optimized_pipeline_score, tpot.score(Xtest, Ytest)])
scoreDf = pd.DataFrame(scores, columns = ['Training Score', 'Testing Score'])
scoreDf.to_csv(outPath + '/scores/scores_' + seed + '.tsv', sep='\t')

file = open(outPath + '/feature_sets.txt', 'a')
file.write(seed + '\t' + tpot.fitted_pipeline_.steps[0][1].sel_subset_name + '\n')
file.close()

model = tpot.fitted_pipeline_
perm = PermutationImportance(model, n_iter=10, random_state=42+int(seed)).fit(Xtest, Ytest)
feat_imp = pd.DataFrame({'feat': list(Xtest), 'score': perm.feature_importances_})
feat_imp.to_csv(outPath + '/scores/featureImp_' + seed + '.tsv', sep='\t', index=False)
