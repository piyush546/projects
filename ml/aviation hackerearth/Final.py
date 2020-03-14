# Modules for preprocessing
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt 

# To load the training data and testing data
train_data  = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# To encode the categorical column
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
train_data["Severity"] =  label_encoder.fit_transform(train_data["Severity"])

# To seperate features and labels
train_features = train_data.iloc[:, 1:].values
train_label = train_data.iloc[:, 0].values


# feature extraction
import statsmodels.api as sm
features_obj = train_features
# To add the constant 
features_obj = sm.add_constant(features_obj)

while (True):
    regressor_OLS = sm.OLS(endog = train_label,exog =features_obj).fit()
    p_values = regressor_OLS.pvalues
    if p_values.max() > 0.05 :
        # deleting a whole column if required that's why 3rd parameter is 1 means axis=1
        features_obj = np.delete(features_obj, p_values.argmax(),1)
    else:
        break

features_obj = features_obj[:, 1:]


# Model Training phase
from sklearn.tree import DecisionTreeClassifier
DTClass = DecisionTreeClassifier()

# Bagging is used as dataset is small
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
num_trees = 100
model = BaggingClassifier(base_estimator=DTClass, n_estimators=num_trees, random_state=seed)
model.fit(features_obj, train_label)



# Prediction or Testing phase
test  = test_data.iloc[:, [0,1, 3,4, 6]]
pred = model.predict(test)
test["Severity"] = label_encoder.inverse_transform(pred)
submission_data = test.loc[:, ["Severity"]]
submission_data["Accident_ID"] =  test_data["Accident_ID"]
submission_data = submission_data.loc[:, ["Accident_ID","Severity"]]
submission_data.to_csv("predictions5.csv", index=False)

