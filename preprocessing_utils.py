import time
import numpy as np
import pandas as pd
import tensorflow as tf
import json
from tensorflow.keras.layers.experimental import preprocessing


# Load dataset in .csv format 
def load_data(csv_data):
    """ Load dataset in .csv format.
    
        Input:
            csv_data: string
            
        Return:
            df: pd.DataFrame
        
        Example:
            load_data('joined_new.csv')
    """
    
    print('Loading data...')
    df = pd.read_csv(csv_data)
    raw_dataset_length = len(df)
    print('Data loaded! The length of the raw dataframe is', raw_dataset_length)
    
    return df


def calculate_datetime(feature, reference, df):
    """ Calculate datetime by subtracting the reference value.
    
        Input:
            feature: string
            reference: string
            df: pd.DataFrame
            
        Return:
            df: pd.DataFrame
            
        Example:
            calculate_datetime(feature, reference, df)
    """
    
    df[feature] = df[feature] - df[reference]
    
    return df
    
    
def set_reference_date(date):
    """ Set value that represents reference date.
        
        Input:
            date: string
            
        Return:
            pandas._libs.tslibs.timestamps.Timestamp
        
        Example:
            set_reference_date(date)
    """
    
    return pd.to_datetime(date)
    
    
def to_datetime(column, df):
    """ Cast dataframe column to datetime.
    
        Input:
            column: string
            df: pd.DataFrame
            
        Return:
            df: pd.DataFrame
            
        Example:
            to_datetime(column, df)
    """
    
    df[column] = pd.to_datetime(df[column])
    
    return df
    
    
def convert_to_datetime(columns, df):
    """ Cast list of dataframe columns to datetime.
    
        Input:
            columns: list of strings
            df: pd.DataFrame
            
        Return:
            df: pd.DataFrame
            
        Example:
            convert_to_datetime(columns, df)
    """
    
    for column in columns:
        df = to_datetime(column, df)
    
    return df


def to_categorical(column, df):
    """ Cast dataframe column to categorical(integer) values. 
    
        Input:
            column: string
            df: pd.DataFrame
            
        Return:
            df: pd.DataFrame
            
        Example:
            to_categorical(column, df)
    """
    
    df.loc[:, column] = pd.Categorical(df[column])
    df.loc[:, column] = df[column].cat.codes + 1
    
    return df


def convert_to_categorical(columns, df):
    """ Cast list of dataframe columns to categorical(integer) values.
    
        Input:
            columns: list of strings
            df: pd.DataFrame
            
        Return:
            df: pd.DataFrame
            
        Example:
            convert_to_categorical(columns, df)
    """
    
    for column in columns:
        df = to_categorical(column, df)
        
    return df


def map_and_save_target(target_variable, df, ddf, path):
    """ Map transformation of target variable from original to transformed dataframe and vice-versa.
        Save mappings in json format.
        
        Input:
            target_variable: string
            df: pd.DataFrame
            ddf: pd.DataFrame
            path: string
            
        Return:
            None (void function)
            
        Example:
            map_and_save_target(target_variable, df, ddf, path)
    """
    
    # Make key-value pairs for mapping target variable 
    category_dict = dict(zip(df[target_variable], ddf[target_variable]))
    inv_category_dict = dict(zip(ddf[target_variable], df[target_variable]))
    
    cat_path = path + 'cat_dict_' + target_variable + '.txt'
    inv_cat_path = path + 'inv_cat_dict_' + target_variable + '.txt'
    
    mappings = {cat_path : category_dict, inv_cat_path : inv_category_dict}
    
    # Save mappings
    for i in mappings.items():
        with open(i[0], 'w') as f:
            json.dump(i[1], f)
            
    

def transform_dataframe(df, target_variable, problem, hashed_features, datetime_columns, numeric_columns, reference_date, dict_path):
    ''' Transform original dataframe to dataframe ready for compressing and training.

        Input:
            df: pd.DataFrame
            target_variable: string
            problem: integer
            hashed_features: list of strings
            datetime_columns: list of strings
            numeric_columns: list of strings
            dict_path: string
            
        Return:
            ddf: pd.DataFrame
            
        Example:
            transform_dataframe(df, target_variable, problem, hashed_features, datetime_columns, numeric_columns, reference_date, dict_path)
            
        0 = classification
        1 = binary classification
        2 = regression
    '''
    
    all_columns = hashed_features + datetime_columns + numeric_columns 
    all_columns.append(target_variable)
    
    ddf = df[all_columns]
    
    ddf.loc[:, 'reference_date'] = set_reference_date(reference_date)
    
    ddf = convert_to_datetime(datetime_columns, ddf)
    
    ddf = convert_to_categorical(hashed_features, ddf)
    
    ddf = calculate_datetime('rfq_created_on', 'reference_date', ddf)
    ddf = calculate_datetime('rfq_dealine_date', 'rfq_created_on', ddf)
    ddf = calculate_datetime('registration_date', 'reference_date', ddf)
    ddf = calculate_datetime('item_created', 'rfq_created_on', ddf)
    
    if problem == 0:
        print('Features ready, preparing the target variable for classification...')
        ddf = to_categorical(target_variable, ddf)

        map_and_save_target(target_variable, df, ddf, dict_path)
        
    elif problem == 1:
        print('Features ready, target variable is prepared for binary classification...')
        
    elif problem == 2:
        
        if target_variable in ['sent_date', 'sent_datetime']:
            print('Features ready, preparing the target variable for regression...')
            ddf = to_datetime(target_variable, ddf)
            ddf[target_variable] = ddf[target_variable] - ddf['rfq_created_on']
            
        elif target_variable == 'created':
            print('Features ready, preparing the target variable for regression...')
            ddf = to_datetime(target_variable, ddf)
            ddf['created'] = ddf[target_variable] - ddf['reference_date']
            
        else:
            print('Features ready, target variable is prepared for regression...')

    print('Dataframe transformed!')
    
    return ddf


# Get number of distinct values in each feature and number of classes
def dataset_measure(ddf, target_variable, problem, hashed_features, var_path):
    ''' Get number of distinct values in each feature and number of classes.
    
        Input:
            ddf: pd.DataFrame
            target_variable: string
            problem: integer
            hashed_features: list of strings
            var_path: string
            
        Return:
            cardinalities: list of integers
            n_classes: integer
            
        Example: 
            dataset_measure(ddf, target_variable, problem, hashed_features, var_path)
        
        0 = classification
        1 = binary classification
        2 = regression
    '''
    
    cardinalities = []
    for i in hashed_features:
        cardinalities.append(len(set(ddf[i])))
    
    n_classes = np.NaN
        
    if problem == 1:
        n_classes = 2
        
    elif problem == 2:
        n_classes = 0
        
    else:
        n_classes = len(set(ddf[target_variable]))
        
    print('Saving variables for inference...')
    cardinalities_save_path = var_path + 'cardinalities_' + target_variable + '.npy'
    np.save(cardinalities_save_path, cardinalities)
    
    n_classes_save_path = var_path + 'n_classes_' + target_variable + '.npy'
    np.save(n_classes_save_path, n_classes)
    print('Variables for inference saved!')
    
    return cardinalities, n_classes
    


def fillna_with_value(ddf, columns, value):
    ''' Fills missing values with numeric values.
        
        Input:
            ddf: pd.DataFrame
            columns: list of strings
            value: string
            
        Return:
            ddf: pd.DataFrame
        
        Example: 
            fillna_with_value(ddf, columns, value)
    '''
    
    for col in columns:
        if value == 'mean':
            ddf[col] = ddf[col].fillna(ddf[col].mean())
        elif value == 'median':
            ddf[col] = ddf[col].fillna(ddf[col].median())
        else:
            ddf[col] = ddf[col].fillna(int(value))
            
    return ddf

            
def reshape_hashed(data_mode, hashed_features):
    ''' Reshape hashed features and store them in a dataframe.
    
        Input:
            data_mode: pd.DataFrame
            hashed_features: list of strings
            
        Return:
            categorical_variables_df: pd.DataFrame
            
        Example: 
            reshape_hashed(data_mode, hashed_features)
    '''
    
    categorical_variables_df = pd.DataFrame()
    for i in hashed_features:
        cat = np.array(data_mode[i])
        categorical_variables_df[i] = cat
    
    return categorical_variables_df



def reshape_datetime(data_mode, datetime_columns):
    ''' Reshape datetime columns and store them in a dataframe.
        
        Input:
            data_mode: pd.DataFrame
            datetime_columns: list of strings
            
        Return:
            datetime_variables_df: pd.DataFrame
            
        Example:
            reshape_datetime(data_mode, datetime_columns)
    '''
    
    datetime_variables_df = pd.DataFrame()
    for i in datetime_columns:
        col = data_mode[i].values.astype(np.int64) // 10 ** 11
        datetime_variables_df[i] = col
        
    return datetime_variables_df



def dataframe_to_numpy(data_mode, target_variable, problem, hashed_features, datetime_columns, numeric_columns):
    """ Transforms variables from dataframe into numpy array and adjusts
        it for the input to tf.Dataset in form of tensor slices.
        
        Input: 
            data_mode: pd.DataFrame
            target_variable: string
            problem: integer
            hashed_features: list of strings
            datetime_columns: list of strings
            numeric_columns: list of strings
            
        Return:
            categorical_variables: np.ndarray
            train_x: np.ndarray
            train_y: np.ndarray
            
        Example:
            dataframe_to_numpy(data_mode, target_variable, problem, hashed_features, datetime_columns, numeric_columns)
            
    - data_mode: values can be 'train' or 'test'
    - target_variable: it can be adjusted to some target variable, e.g. 'abc'
    - problem: can be 'classification' for multi-class classification, 'binary_classification'
    and regression
    -return: 20 categorical variables and 33 numerical variables
    """
    
    categorical_variables_df = reshape_hashed(data_mode, hashed_features)
    
    datetime_variables_df = reshape_datetime(data_mode, datetime_columns)

    numerical_features_df = data_mode[numeric_columns]

    train_x = pd.concat([datetime_variables_df, numerical_features_df], axis=1)
    train_x = np.array(train_x)
    
    categorical_variables = np.array(categorical_variables_df)

    if problem in [0, 1]:
        train_y = data_mode[target_variable].values.astype(np.int64)
        train_y = train_y.reshape(-1, 1)
    elif problem == 2:
        if target_variable in ['sent_date', 'created', 'sent_datetime']:
            train_y = data_mode[target_variable].values.astype(np.int64)
            train_y = train_y // 10 ** 11 * 1.15741e-3
            train_y = train_y.reshape(-1, 1)
            train_y = train_y.astype(int)
        else:
            train_y = data_mode[target_variable].values.astype(np.int64)
            train_y = train_y.reshape(-1, 1)

    return categorical_variables, train_x, train_y



def normalize_data(x_train, x_test):
    """ Data normalization for numerical input features. It requires inport
        from tensorflow.keras.layers.experimental import preprocessing.
        
        Input:
            x_train: np.ndarray
            x_test: np.ndarray
            
        Return:
            x_train_: tf.Tensor
            x_test_: tf.Tensor
            
        Example:
            normalize_data(x_train, x_test)
    """
    
    print('Normalizing data...')
    normalizer = preprocessing.Normalization()
    normalizer.adapt(x_train)
    x_train_ = normalizer(x_train)
    x_test_ = normalizer(x_test)
    print('Data normalization successful!')
    
    return x_train_, x_test_

    

def clean_data(ddf, columns_to_fillna_with_mean, columns_to_fillna_with_median, columns_to_fillna_with_0, columns_to_fillna_with_1):
    ''' Column imputation with different values and methods.
    
        Input:
            ddf: pd.DataFrame
            columns_to_fillna_with_mean: list of strings 
            columns_to_fillna_with_median: list of strings
            columns_to_fillna_with_0: list of strings
            columns_to_fillna_with_1: list of strings
            
        Return:
            ddf: pd.DataFrame
            
        Example:
            clean_data(ddf, columns_to_fillna_with_mean, columns_to_fillna_with_median, columns_to_fillna_with_0, columns_to_fillna_with_1)
    '''
    
    ddf = fillna_with_value(ddf, columns_to_fillna_with_mean, 'mean')
    ddf = fillna_with_value(ddf, columns_to_fillna_with_median, 'median')
    ddf = fillna_with_value(ddf, columns_to_fillna_with_0, 0)
    ddf = fillna_with_value(ddf, columns_to_fillna_with_1, 1)
    #ddf['registration_date'] = ddf['registration_date'].fillna(ddf['registration_date'].median()) #TODO
    ddf.dropna(inplace=True)
    
    return ddf
    

    
def preprocess_data(df, target_variable, problem, hashed_features, datetime_columns, numeric_columns, reference_date, split_frac, dict_path, var_path,                                             columns_to_fillna_with_mean, columns_to_fillna_with_median, columns_to_fillna_with_0, columns_to_fillna_with_1):
    ''' Main function for data preprocessing. It contains all helpfull function calls to get prepared features.
        
        Input:
            df: pd.DataFrame
            target_variable: string
            problem: int
            hashed_features: list of strings
            datetime_columns: list of strings
            numeric_columns: list of strings
            reference_date: string
            split_frac: float
            dict_path: string
            var_path: string
            columns_to_fillna_with_mean: list of strings
            columns_to_fillna_with_median: list of strings
            columns_to_fillna_with_0: list of strings
            columns_to_fillna_with_1: list of strings
            
        Return:
            train_X: tensorflow.python.framework.ops.EagerTensor
            train_cat: np.ndarray
            train_y: np.ndarray
            test_X: tensorflow.python.framework.ops.EagerTensor
            test_cat: np.ndarray
            test_y: np.ndarray
            
        Example:
            preprocess_data(df, target_variable, problem, hashed_features, datetime_columns, numeric_columns, reference_date, split_frac, dict_path, var_path,                                             columns_to_fillna_with_mean, columns_to_fillna_with_median, columns_to_fillna_with_0, columns_to_fillna_with_1)
    '''
    
    ddf = transform_dataframe(df, target_variable, problem, hashed_features, datetime_columns, numeric_columns, reference_date, dict_path)

    cardinalities, n_classes = dataset_measure(ddf, target_variable, problem, hashed_features, var_path)

    ddf = clean_data(ddf, columns_to_fillna_with_mean, columns_to_fillna_with_median, columns_to_fillna_with_0, columns_to_fillna_with_1)

    train = ddf.sample(frac=split_frac, random_state=1234)  # train-test split
    test = ddf.drop(train.index)

    dataset_length = len(ddf)
    print('The length of the dataset after preprocessing: {}.'.format(dataset_length))

    train_cat, train_X, train_y = dataframe_to_numpy(train, target_variable, problem, hashed_features, datetime_columns, numeric_columns)

    test_cat, test_X, test_y = dataframe_to_numpy(test, target_variable, problem, hashed_features, datetime_columns, numeric_columns)
    
    train_X, test_X = normalize_data(train_X, test_X)
    
    return train_X, train_cat, train_y, test_X, test_cat, test_y





    
    
