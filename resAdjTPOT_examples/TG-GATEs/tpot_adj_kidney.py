import pandas as pd
import numpy as np
import sys
from tpot.builtins import resAdjPredefinedSplits
from tpot.builtins import resAdjR2Scorer
from tpot.config import resAdj_regressor_config_dict
from tpot import TPOTRegressor
import re
import eli5
from eli5.sklearn import PermutationImportance

seed = sys.argv[1]
inTrainFile = sys.argv[2]
inTestFile = sys.argv[3]
outPath = sys.argv[4] 
n_gen = int(sys.argv[5])
n_pop = int(sys.argv[6])
tissue = sys.argv[7]

Xtrain = pd.read_csv(inTrainFile, sep='\t', index_col='BARCODE')
Ytrain = Xtrain['adjY']
Xtrain.drop(['adjY'], axis=1,  inplace=True)
Xtest = pd.read_csv(inTestFile, sep='\t', index_col='BARCODE')
Ytest = Xtest['adjY']
Xtest.drop(['adjY'], axis=1,  inplace=True)

reserved = []
for col in Xtrain.columns:
    if re.match(r'^indicator', str(col)) or re.match(r'^adjY', str(col)):
        reserved.append(str(col))

resAdj_regressor_config_dict['tpot.builtins.resAdjTransformer'] ={
    'C': [['DoseOrd', 'SacriOrd', 'COMPOUND_1', 'COMPOUND_2', 'COMPOUND_3', 'COMPOUND_4', 'COMPOUND_5', 'COMPOUND_6']],
    'adj_list': ['adj_list_' + tissue + '.csv']
    
}
resAdj_regressor_config_dict['tpot.builtins.FeatureSetSelector'] ={
    'subset_list': ['fsProbesets2orMoreGenes.csv'],
    'sel_subset': range(154),
    'res_cols': [['DoseOrd', 'SacriOrd', 'COMPOUND_1', 'COMPOUND_2', 'COMPOUND_3', 'COMPOUND_4', 'COMPOUND_5', 'COMPOUND_6'] + reserved]
}

tpot = TPOTRegressor(generations=n_gen, population_size=n_pop, 
                     verbosity=2, cv=resAdjPredefinedSplits(n_splits=5),
                     config_dict=resAdj_regressor_config_dict,
                     template="FeatureSetSelector-resAdjTransformer-Transformer-Regressor",
                     scoring=resAdjR2Scorer, random_state=42 + int(seed))
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
