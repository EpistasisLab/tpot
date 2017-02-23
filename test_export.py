from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

titanic = pd.read_csv('tutorials/data/titanic_train.csv')
titanic.head(5)
titanic.groupby('Sex').Survived.value_counts()
titanic.groupby(['Pclass','Sex']).Survived.value_counts()
id = pd.crosstab([titanic.Pclass, titanic.Sex], titanic.Survived.astype(float))
id.div(id.sum(1).astype(float), 0)


titanic.rename(columns={'Survived': 'class'}, inplace=True)
titanic.dtypes

for cat in ['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked']:
    print("Number of levels in category '{0}': \b {1:2.2f} ".format(cat, titanic[cat].unique().size))

for cat in ['Sex', 'Embarked']:
    print("Levels for catgeory '{0}': {1}".format(cat, titanic[cat].unique()))

titanic['Sex'] = titanic['Sex'].map({'male':0,'female':1})
titanic['Embarked'] = titanic['Embarked'].map({'S':0,'C':1,'Q':2})

titanic = titanic.fillna(-999)
pd.isnull(titanic).any()

from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
CabinTrans = mlb.fit_transform([{str(val)} for val in titanic['Cabin'].values])

CabinTrans

titanic_new = titanic.drop(['Name','Ticket','Cabin','class'], axis=1)

assert (len(titanic['Cabin'].unique()) == len(mlb.classes_)), "Not Equal" #check correct encoding done

titanic_new = np.hstack((titanic_new.values,CabinTrans))

np.isnan(titanic_new).any()

titanic_new[0].size

titanic_class = titanic['class'].values

training_indices, validation_indices = training_indices, testing_indices = train_test_split(titanic.index, stratify = titanic_class, train_size=0.05, test_size=0.95)
training_indices.size, validation_indices.size

tpot = TPOTClassifier(verbosity=2, max_time_mins=2, generations= 5)
print(titanic_new[training_indices])
print(titanic_class[training_indices])
tpot.fit(titanic_new[training_indices], titanic_class[training_indices])

#tpot.score(titanic_new[validation_indices], titanic.loc[validation_indices, 'class'].values)

#tpot.export('tpot_titanic_pipeline.py')
