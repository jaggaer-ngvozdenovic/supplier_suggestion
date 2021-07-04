import time
import numpy as np
import pandas as pd
import tensorflow as tf


def get_example_object_float(data_record):
    ''' Helpful function to get float features and prepare them for .tfrecord serialization.
    
        Input:
            data_record: tensorflow.python.framework.ops.EagerTensor
            
        Return:
            example: tensorflow.core.example.example_pb2.Example
            
        Example:
            get_example_object_float(data_record)
    '''
    
    tf.compat.v1.reset_default_graph()
    
    # Convert individual data into a list of int64 or float or bytes
    float_list1 = tf.train.FloatList(value = data_record)
    

    # Create a dictionary with above lists individually wrapped in Feature
    feature_key_value_pair = {
        'float_list1': tf.train.Feature(float_list = float_list1)
    }

    # Create Features object with above feature dictionary
    features = tf.train.Features(feature = feature_key_value_pair)

    # Create Example object with features
    example = tf.train.Example(features = features)
    
    return example


def get_example_object_int(data_record):
    ''' Helpful function to get integer features and prepare them for .tfrecord serialization.
    
        Input:
            data_record: np.ndarray
            
        Return:
            example: tensorflow.core.example.example_pb2.Example
            
        Example:
            get_example_object_int(data_record) 
    '''
    
    tf.compat.v1.reset_default_graph()
    
    # Convert individual data into a list of int64 or float or bytes
    int_list1 = tf.train.Int64List(value = data_record)
    

    # Create a dictionary with above lists individually wrapped in Feature
    feature_key_value_pair = {
        'int_list1': tf.train.Feature(int64_list = int_list1)
    }

    # Create Features object with above feature dictionary
    features = tf.train.Features(feature = feature_key_value_pair)

    # Create Example object with features
    example = tf.train.Example(features = features)
    
    return example


def compress_tfrecord_float(dataset_float, filename):
    ''' Compress float features into .tfrecord format.
    
        Input:
            dataset_float: tensorflow.python.framework.ops.EagerTensor
            filename: string
            
        Return:
            None (void function)
            
        Example:
            compress_tfrecord_float(dataset_float, filename)
    '''
    
    start = time.time()
    with tf.io.TFRecordWriter(filename) as tfwriter:
        # Iterate through all records
        for data_record in dataset_float:
            example = get_example_object_float(data_record)

            # Append each example into tfrecord
            tfwriter.write(example.SerializeToString())
    end = time.time()
    print(end - start)



def compress_tfrecord_int(dataset_int, filename):
    ''' Compress integer features into .tfrecord format.
    
        Input:
            dataset_int: tensorflow.python.framework.ops.EagerTensor
            filename: string
            
        Return:
            None (void function)
            
        Example:
            compress_tfrecord_int(dataset_int, filename)
    '''
    
    start = time.time()
    with tf.io.TFRecordWriter(filename) as tfwriter:
        # Iterate through all records
        for data_record in dataset_int:
            example = get_example_object_int(data_record)

            # Append each example into tfrecord
            tfwriter.write(example.SerializeToString())
    end = time.time()
    print(end - start)
    
    

# Extract numeric features
def extract_numeric(tfrecord):
    ''' Extract float features from .tfrecord format.
    
        Input:
            tfrecord: string
            
        Return:
            records_num: np.ndarray
            
        Example:
            extract_numeric(tfrecord)
    '''
    
    raw_dataset = tf.data.TFRecordDataset(tfrecord)
    records_num = []
    for raw_record in raw_dataset:
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        records_num.append(example.features.feature['float_list1'].float_list.value)
    records_num = np.array(records_num)
    
    return records_num



# Extract numeric features
def extract_categorical(tfrecord):
    ''' Extract integer features from .tfrecord format.
    
        Input:
            tfrecord: string
            
        Return:
            records_num: np.ndarray
            
        Example:
            extract_numeric(tfrecord)
    '''
    
    raw_dataset = tf.data.TFRecordDataset(tfrecord)
    records_cat = []
    for raw_record in raw_dataset:
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        records_cat.append(example.features.feature['int_list1'].int64_list.value)
    records_cat = np.array(records_cat)
    
    return records_cat



def compress_tfrecord(train_num, test_num, train_cat, test_cat, train_target, test_target, target_variable, path):
    ''' Compress numeric and float features and target variable as .tfrecord files.
    
        Input:
            train_num: tensorflow.python.framework.ops.EagerTensor
            test_num: tensorflow.python.framework.ops.EagerTensor
            train_cat: np.ndarray
            test_cat: np.ndarray
            train_target: np.ndarray
            test_target: np.ndarray
            target_variable: string
            path: string
            
        Return:
            None (void function)
            
        Example:
            compress_tfrecord(train_num, test_num, train_cat, test_cat, train_target, test_target, target_variable, path)
    '''
        
    # Compress numeric and target features into TFRecord files
    compress_tfrecord_float(train_num, path + 'train_num_' + target_variable + '.tfrecord')
    compress_tfrecord_float(test_num, path + 'test_num_' + target_variable + '.tfrecord')
    print('Numerical features are saved as TFRecord files!')

    compress_tfrecord_int(train_target, path + 'train_target_' + target_variable + '.tfrecord')
    compress_tfrecord_int(test_target, path + 'test_target_' + target_variable + '.tfrecord')
    print('Target variable is saved as TFRecord file!')
    
    # Compress categorical features into TFRecord files
    compress_tfrecord_int(train_cat, path + 'train_cat_' + target_variable + '.tfrecord')
    compress_tfrecord_int(test_cat, path + 'test_cat_' + target_variable + '.tfrecord')
    print('Categorical features are saved as TFRecord files!')
    
    

def extract_features(path, target_variable):
    """ Extracts all features and target variable from .tfrecord files
        by calling custom functions.
        
        Input: 
            path: string (where files are located)
            target_variable: string (name of a column to be used for prediction)
            
        Return:
            num_train_records: np.ndarray
            cat_train_records: np.ndarray
            target_train_records: np.ndarray
            num_test_records: np.ndarray
            cat_test_records: np.ndarray
            target_test_records: np.ndarray
            
        Example: 
            extract_features('./saved_features/', 'abc')
    """
        
    # Extract numeric features for training
    num_train_records = extract_numeric(path + 'train_num_' + target_variable + '.tfrecord')
    
    # Extract categorical features for training
    cat_train_records = extract_categorical(path + 'train_cat_' + target_variable + '.tfrecord')
    
    # Extract target variable for training
    target_train_records = extract_categorical(path + 'train_target_' + target_variable + '.tfrecord')
    
    # Extract numeric features for testing
    num_test_records = extract_numeric(path + 'test_num_' + target_variable + '.tfrecord')
    
    # Extract categorical features for testing
    cat_test_records = extract_categorical(path + 'test_cat_' + target_variable + '.tfrecord')
    
    # Extract target variable for testing
    target_test_records = extract_categorical(path + 'test_target_' + target_variable + '.tfrecord')
        
    return num_train_records, cat_train_records, target_train_records, num_test_records, cat_test_records, target_test_records