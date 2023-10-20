import torch
from torch.autograd import Variable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
from logistic_regression_model import LogisticRegressionModel
import sklearn.preprocessing as slp
from torch.utils import data
from torch.backends import cudnn
from dataset import Dataset
from sklearn.model_selection import train_test_split

def gen_submission():

    # === Data fetch ===


    df_submission = pd.read_csv('data/submission_data.csv')
    starting_id = df_submission.pop('PassengerId')[0]


    # === Data preprocessing ===


    # Get Revelant Data
    x_features_to_binary = df_submission[['Sex','Embarked']]
    x_features_no_change = df_submission[['Pclass','Age','SibSp','Parch']]


    # Encode Data
    label_encoder = slp.LabelEncoder()
    one_hot_encoder = slp.OneHotEncoder()

    x_targets_without_extra_column = one_hot_encoder.fit_transform(x_features_to_binary).toarray()
    # Add dummy columns to either testing or training data because its shape along the feature axis must be equivalent to one another
    nan_array = np.full(shape=(1, x_targets_without_extra_column.shape[0]), fill_value=np.nan)
    x_targets = np.transpose(np.concatenate((np.transpose(x_targets_without_extra_column), nan_array)))


    # Combine & Finalize Data
    x_merged = np.concatenate((np.transpose(x_targets), np.transpose(x_features_no_change.to_numpy())))
    x_submit_final = Variable(torch.Tensor(np.transpose(x_merged)))
    x_submit_final[torch.isnan(x_submit_final)] = 0


    # === Load Model ===


    model = torch.load('model.pt')


    # === Predict Values & Submit to File ===

    df = pd.DataFrame(columns=['PassengerId', 'Survived'])
    pas_num = starting_id
    for row in x_submit_final:
        pred_y = model(row)
        rounded_pred = round(pred_y.item())
        df.loc[len(df)] = [pas_num, rounded_pred]
        pas_num+=1

    df.to_csv('data/submission.csv', index = False)