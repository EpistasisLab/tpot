
# coding: utf-8

# In[1]:

import pandas as pd
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split


# In[2]:

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')


# In[3]:

combined_set = pd.concat([train_data, test_data])
combined_set['combined_var'] = (combined_set.hair_length * .40) + (combined_set.has_soul * .40)

# Replace categorical variables with numbers
def label_encoding(df, col):
    label_map = { key: float(n) for n, key in enumerate(df[col].unique()) }
    label_reverse_map = { label_map[key]: key for key in label_map }
    df[col] = df[col].apply(lambda x: label_map[x])
    return df, label_map, label_reverse_map

combined_set = pd.get_dummies(combined_set, columns=['color'])
combined_set

train_set = combined_set[:len(train_data.index)]
test_set = combined_set[len(train_data.index):]


# In[4]:

train_cols = ['combined_var', 'rotting_flesh', 'bone_length', 'has_soul', 'hair_length']
target_var = ['type']
selected_cols = train_cols + target_var


# In[5]:

train_set, type_label_map, type_label_reverse_map = label_encoding(train_set, 'type')


# In[6]:

p_train,val = train_test_split(train_set, train_size=.75, test_size=.25)


# In[7]:

p_train.shape, val.shape


# In[8]:

p_train[train_cols].head().values


# In[9]:

tpot = TPOTClassifier(verbosity=3, generations = 5)



# In[12]:
#print(pd.np.array(p_train[train_cols]))
print(pd.np.array(p_train[target_var]))
print(pd.np.array(p_train[target_var]).ravel())

tpot.fit(pd.np.array(p_train[train_cols]), pd.np.array(p_train[target_var]).ravel())


# In[21]:

p_train[train_cols].head()


# In[23]:

p_train[target_var].head()


# In[ ]:
