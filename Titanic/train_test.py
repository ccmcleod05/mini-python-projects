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

def train_test():

    # === Data fetch ===


    df_all = pd.read_csv('data/input_data.csv')
    df_train, df_test = train_test_split(df_all)


    # === Data preprocessing ===


    # Get Revelant Data
    y_labels = df_train.pop('Survived')
    x_features_to_binary = df_train[['Sex','Embarked']]
    x_features_no_change = df_train[['Pclass','Age','SibSp','Parch']]


    y_test_labels = df_test.pop('Survived')
    x_test_features_to_binary = df_test[['Sex','Embarked']]
    x_test_features_no_change = df_test[['Pclass','Age','SibSp','Parch']]


    # Encode Data
    label_encoder = slp.LabelEncoder()
    one_hot_encoder = slp.OneHotEncoder()

    y_targets = label_encoder.fit_transform(y_labels)
    x_targets_without_extra_column = one_hot_encoder.fit_transform(x_features_to_binary).toarray()

    y_targets_expanded = np.expand_dims(y_targets, axis=1)


    y_test_targets = label_encoder.fit_transform(y_test_labels)
    x_test_targets_without_extra_column = one_hot_encoder.fit_transform(x_test_features_to_binary).toarray()

    # Add dummy columns to either testing or training data because its shape along the feature axis must be equivalent to one another
    nan_array = None
    x_targets = x_targets_without_extra_column
    x_test_targets = x_test_targets_without_extra_column
    # Get info on which dataset has more feature columns than the other and add accordingly
    feature_diff = x_test_targets.shape[1]-x_targets.shape[1]
    if feature_diff > 0: # pos
        nan_array = np.full(shape=(1, y_targets.shape[0]), fill_value=np.nan)
        for i in range(0, abs(feature_diff)):    
            x_targets = np.transpose(np.concatenate((np.transpose(x_targets), nan_array)))
            i+=1
    if feature_diff < 0: # neg
        nan_array = np.full(shape=(1, y_test_targets.shape[0]), fill_value=np.nan)
        for i in range(0, abs(feature_diff)):
            x_test_targets = np.concatenate((np.transpose(x_test_targets), nan_array))
            i+=1
    else:
        x_test_targets = np.transpose(x_test_targets_without_extra_column)

    y_test_targets_expanded = np.expand_dims(y_test_targets, axis=1)


    # Combine & Finalize Data
    y_train_final = Variable(torch.Tensor(y_targets_expanded))

    x_merged = np.concatenate((np.transpose(x_targets), np.transpose(x_features_no_change.to_numpy())))
    x_train_final = Variable(torch.Tensor(np.transpose(x_merged)))
    x_train_final[torch.isnan(x_train_final)] = 0


    y_test_final = Variable(torch.Tensor(y_test_targets_expanded))
    x_test_merged = np.concatenate((x_test_targets, np.transpose(x_test_features_no_change.to_numpy())))
    x_test_final = Variable(torch.Tensor(np.transpose(x_test_merged)))
    x_test_final[torch.isnan(x_test_final)] = 0


    # === Data Train & Validate ===


    # Create Model
    model = LogisticRegressionModel()


    # Create Loss Function & Optimizer (Criterion is the value outputted by the loss function? So then is the predicted - actual the loss or criterion?)
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(params=model.parameters(), lr=.00009)


    # Accuracy Measurement Function
    def measure_acc(y_pred, local_labels):
        y_rounded = torch.round(y_pred)

        correct_pred_count = (y_rounded == local_labels).sum()
        acc = correct_pred_count/local_labels.shape[0]
        acc = torch.round(acc * 100)
            
        return acc


    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    cudnn.benchmark = True

    # Parameters
    params = {'batch_size': 64,
            'shuffle': True,
            'num_workers': 0}
    max_epochs = 500

    # Generators
    training_set = Dataset(x_train_final, y_train_final)
    training_generator = data.DataLoader(training_set, **params)

    validation_set = Dataset(x_test_final, y_test_final)
    validation_generator = data.DataLoader(validation_set, **params)

    # Training
    for epoch in range(max_epochs):

        epoch_loss = 0
        epoch_acc = 0

        for local_batch, local_labels in training_generator:
            # Transfer to GPU
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)

            pred_y = model(local_batch)
                
            loss = criterion(pred_y, local_labels)
            acc = measure_acc(pred_y, local_labels)
            
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Increment total loss and acc of epoch
            epoch_loss += loss
            epoch_acc += acc

        # Print Results of Epoch
        print(f't: epoch {epoch}, loss {epoch_loss/len(training_generator)}, acc {epoch_acc/len(training_generator)}')

    # Validation
    for epoch in range(max_epochs):
            
        epoch_loss = 0
        epoch_acc = 0

        for local_batch, local_labels in validation_generator:
            # Transfer to GPU
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)

            pred_y = model(local_batch)
                    
            loss = criterion(pred_y, local_labels)
            acc = measure_acc(pred_y, local_labels)
                
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Increment total loss and acc of epoch
            epoch_loss += loss
            epoch_acc += acc

        # Print Results of Epoch
        print(f'v: epoch {epoch}, loss {epoch_loss/len(validation_generator)}, acc {epoch_acc/len(validation_generator)}')


    # === Save Model ===

    torch.save(model, 'model.pt')