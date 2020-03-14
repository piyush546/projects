#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 16:33:01 2020

@author: piyush
"""

# Data loading step
import pandas as pd
import numpy as np


train_transaction_data = pd.read_csv("train_transaction.csv")
train_identity_data = pd.read_csv("train_identity.csv")
training_data = train_transaction_data.set_index('TransactionID').join(train_identity_data.set_index('TransactionID'))


# to check the memory usage of a dataset loaded
training_data.info(memory_usage='deep')


# Data wrangling step
def data(dataset, values, limit):
    index = list(values<limit)
    return dataset.iloc[:,index]

val = training_data.isnull().sum(axis=0).values


training_data = data(training_data ,val, 60000)

def miss(dataset):
    dataset = dataset.fillna(dataset.max())
    dataset = dataset.replace(np.nan, "Missing")
    return dataset

training_data = miss(training_data)


def reduce_mem_usage(props):
    start_mem_usg = props.memory_usage().sum() / 1024**2 
    print("Memory usage of properties dataframe is :",start_mem_usg," MB")
    NAlist = [] # Keeps track of columns that have missing values filled in. 
    for col in props.columns:
        if props[col].dtype != object:  # Exclude strings
            
            # Print current column type
            print("******************************")
            print("Column: ",col)
            print("dtype before: ",props[col].dtype)
            
            # make variables for Int, max and min
            IsInt = False
            mx = props[col].max()
            mn = props[col].min()
            
            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(props[col]).all(): 
                NAlist.append(col)
                props[col].fillna(mn-1,inplace=True)  
                   
            # test if column can be converted to an integer
            asint = props[col].fillna(0).astype(np.int64)
            result = (props[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True

            
            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        props[col] = props[col].astype(np.uint8)
                    elif mx < 65535:
                        props[col] = props[col].astype(np.uint16)
                    elif mx < 4294967295:
                        props[col] = props[col].astype(np.uint32)
                    else:
                        props[col] = props[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        props[col] = props[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        props[col] = props[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        props[col] = props[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        props[col] = props[col].astype(np.int64)    
            # Make float datatypes 32 bit
            else:
                props[col] = props[col].astype(np.float32)
                
                
            # Print new column type
            print("dtype after: ",props[col].dtype)
            print("******************************")
        else:
            props[col] = props[col].astype('category')
    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = props.memory_usage().sum() / 1024**2 
    print("Memory usage is: ",mem_usg," MB")
    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
    return props, NAlist




training_data, NAlist = reduce_mem_usage(training_data)
print("_________________")
print("")
print("Warning: the following columns have missing values filled with 'df['column_name'].min() -1': ")
print("_________________")
print("")
print(NAlist)

# Handling the categorical columns
from sklearn.preprocessing import LabelEncoder
encoders = []
cat_col = [x for x in training_data.columns if training_data[x].dtype.name == "category"]
for val in cat_col:
     obj= LabelEncoder()
     training_data[val] = obj.fit_transform(training_data[val]).astype(np.uint8)
     encoders.append(obj)



features = training_data.drop("isFraud", axis=1)
labels = training_data["isFraud"]


test_transaction_data = pd.read_csv("test_transaction.csv")
test_identity_data = pd.read_csv("test_identity.csv")
testing_data = test_transaction_data.set_index('TransactionID').join(test_identity_data.set_index('TransactionID'))


testing_data.info(memory_usage='deep')
print(test_transaction_data.head())
print(test_transaction_data.isnull().sum(axis=0).values)


col = list(training_data.columns)
col.remove("isFraud")
testing_data = testing_data.loc[:,col]
testing_data = miss(testing_data)
testing_data = testing_data.replace("scranton.edu", "gmail.com")
testing_data, NAlist_2 = reduce_mem_usage(testing_data)

print("_________________")
print("")
print("Warning: the following columns have missing values filled with 'df['column_name'].min() -1': ")
print("_________________")
print("")
print(NAlist_2)



for index, val4 in enumerate(cat_col):
    testing_data[val4] = encoders[index].transform(testing_data[val4]).astype(np.uint8)


# Model training and testing phase
from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier(n_estimators=200, random_state=0)
clf.fit(features, labels)
clf.score(features, labels)


real_labels_test = clf.predict(testing_data)


# Creating the required csv file
result_dataset = pd.DataFrame()
result_dataset["TransactionID"] = testing_data.index.values

result_dataset["isFraud"] = real_labels_test
result_dataset.to_csv("submission.csv", index=False)


