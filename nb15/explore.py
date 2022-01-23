import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)

# DONE: Need to one hot encode the following columns: proto, service, state

# Read csv using pandas
df = pd.read_csv('nb15/data/UNSW_NB15_training-set.csv')
# Prepare data
df = df.drop(columns=df.columns[0], axis=1)
df.loc[df['service'] == '-', 'service'] = 'none'
# One-hot encode catagorical columns
df = pd.get_dummies(df, columns=['proto', 'service', 'state'], prefix=['proto', 'service', 'state'])
print(df.dtypes)
print(df.columns)
# print(df.head())

df_attack_cat = df['attack_cat']
df_label = df['label']

print(type(df_attack_cat))

df = df.drop(columns=['label', 'attack_cat'], axis=1)
df = (df-df.min())/(df.max()-df.min())
x_train = df.to_numpy()
print(x_train.shape)