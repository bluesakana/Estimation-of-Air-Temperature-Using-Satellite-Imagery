import numpy as np
import pandas as pd

import tensorflow as tf

from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error

import glob

# Identity Transformer for particular columns
identity_transformer = FunctionTransformer(lambda x: x, feature_names_out='one-to-one')

# One-Hot Encoding
onehot_transformer = OneHotEncoder(handle_unknown='ignore') # set unknown cat to all zeros

def model_performance(model, X_train, X_test, y_train, y_test):
    '''
    Function that computes and displays performance metrics of RMSE and MAE
    '''
    print('----------Mean Absolute Error----------')
    print(f'MAE Train: {mean_absolute_error(y_train, model.predict(X_train)):.2f}')
    print(f'MAE Test: {mean_absolute_error(y_test, model.predict(X_test)):.2f}')
    
    print('--------Root Mean Squared Error--------')
    print(f'RMSE Train: {mean_squared_error(y_train, model.predict(X_train), squared=False):.2f}')
    print(f'RMSE Test: {mean_squared_error(y_test, model.predict(X_test), squared=False):.2f}')
    
    
def preproc_sen2all(df):
    '''
    Function that takes in the 'image' dataframe, preprocesses it into tensors for model that predicts on sen2 all data 
    '''
    
    # normalise to RGB values
    temp = df.map(lambda x: x[0]*255)
    
    # convert dtype to float32
    temp = temp.map(lambda x: x.astype('float32'))
    
    # convert to tensors
    temp = temp.map(lambda x: tf.convert_to_tensor(x)) 
    
    # convert to tensor iterator
    temp = tf.data.Dataset.from_tensor_slices(temp.to_list())
    
    # define batch size, as the model was trained in batches
    temp = temp.batch(batch_size=1)
    
    return temp
    
    
def preproc_sen2rgb(df):
    '''
    Function that takes in the 'image' dataframe, preprocesses it into tensors for model that predicts on sen2 rgb data 
    '''
    
    # normalise to RGB values
    temp = normalise_rgb(df)
    
    # convert to tensors
    temp = temp.map(lambda x: tf.convert_to_tensor(x)) 
    
    # convert to tensor iterator
    temp = tf.data.Dataset.from_tensor_slices(temp.to_list())
    
    # define batch size, as the model was trained in batches
    temp = temp.batch(batch_size=1)
    
    return temp


def get_model_history(file_path):
    # read in all the json files
    file_list = glob.glob(file_path)
    file_list.sort()
    performance_df = []

    for file in file_list:
        performance = pd.read_json(file)
        performance_df.append(performance)

    # concat all the jsons into a single dataframe
    performance_df = pd.concat(performance_df, ignore_index=True)
    
    # add in the indexing for the epoch
    performance_df['epoch'] = np.arange(1, len(performance_df)+1)
    
    return performance_df