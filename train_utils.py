import time
from datetime import datetime
import json
import os
import argparse # TO-EXPLAIN

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from preprocessing_utils import transform_dataframe, dataset_measure
from compress_and_extract_utils import extract_features
from nn_model import NNModel, build_nn_model

optimizer = tf.keras.optimizers.SGD() # initializes optimizer
train_loss = tf.keras.losses.MeanSquaredLogarithmicError() # initializes train_loss


def compile(problem):
    """ Function that returns compile objects.
    
        Input:
            problem:
            
            if problem = 0 => classification
            if problem = 1 => binary classification
            if problem = 2 => regression
            
        Return:
            dict('loss_object': loss_object, 'optimizer': optimizer, 'train_loss': train_loss, 'train_metric': train_metric, 
                 'test_loss': test_loss, 'test_metric': test_metric)
                 
            loss_object: tf.keras.losses
            optimizer: tf.keras.optimizers
            train_loss: tf.keras.metrics
            train_metric: tf.keras.metrics
            test_loss: tf.keras.metrics
            test_metric: tf.keras.metrics
            
        Example:
            compile(0)
    """
    
    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
    
    if problem == 2:
        
        loss_object_1 = tf.keras.losses.MeanSquaredError()
        loss_object_2 = tf.keras.losses.MeanAbsoluteError()
        loss_object = [loss_object_1, loss_object_2]
        
        optimizer = tf.keras.optimizers.Nadam(learning_rate=0.0001)
        
        train_metric = tfa.metrics.r_square.RSquare('train_r_square', y_shape=(1,))
        
        test_metric = tfa.metrics.r_square.RSquare('test_r_square', y_shape=(1,))
    
    else:
        optimizer = tf.keras.optimizers.Nadam()
        
        if problem == 0:
            loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
            train_metric = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')
            test_metric = tf.keras.metrics.SparseCategoricalAccuracy('test_accuracy')
        elif problem == 1:
            loss_object = tf.keras.losses.BinaryCrossentropy()
            train_metric = tf.keras.metrics.BinaryAccuracy('train_accuracy')
            test_metric = tf.keras.metrics.BinaryAccuracy('test_accuracy')
            
    return {'loss_object': loss_object, 'optimizer': optimizer, 'train_loss': train_loss, 'train_metric': train_metric, 
            'test_loss': test_loss, 'test_metric': test_metric}


    

def train_step(model, compile_params, x_train, y_train, problem, num_layers=3):
    """ Train function for optimizing risk metric.
        
        Input:
            model: nn_model_refactor.NNModel
            compile_params: dict
            x_train: tuple
            y_train: tensorflow.python.framework.ops.EagerTensor
            problem: integer (0 = classification, 1 = binary classification, 2 = regression)
            num_layers: integer
            
        Return:
            None (void function)
            
        Example:
            train_step(model, compile_params, x_train, y_train, problem, num_layers=3)
    """
    
    with tf.GradientTape() as tape:
        predictions = model(x_train, num_layers, training=True)
        if problem == 2:
            loss = compile_params['loss_object'][0](y_train, predictions) + compile_params['loss_object'][1](y_train, predictions)
        else:
            loss = compile_params['loss_object'](y_train, predictions)
    grads = tape.gradient(loss, model.trainable_variables)
    compile_params['optimizer'].apply_gradients(zip(grads, model.trainable_variables))
    compile_params['train_loss'](loss)
    compile_params['train_metric'](y_train, predictions)



# Test function for calculating risk metric        
def test_step(model, compile_params, x_test, y_test, problem, num_layers=3):
    """ Test function for calculating risk metric.
    
        Input:
            model: nn_model_refactor.NNModel
            compile_params: dict
            x_test: tuple
            y_test: tensorflow.python.framework.ops.EagerTensor
            problem: integer
            num_layers: integer
            
        Return:
            None (void function)
            
        Example:
            test_step(model, compile_params, x_test, y_test, problem, num_layers)
    """
    
    predictions = model(x_test, num_layers)
    if problem == 2:
        loss = compile_params['loss_object'][0](y_test, predictions) + compile_params['loss_object'][1](y_test, predictions)
    else:
        loss = compile_params['loss_object'](y_test, predictions)
    compile_params['test_loss'](loss)
    compile_params['test_metric'](y_test, predictions)
    

    
def early_stopping(target_variable, problem, epochs, train_dataset, test_dataset, model, results_file, train_summary_writer, test_summary_writer):
    """ Custom implementation of early stopping method to improve performances of neural network on traininig and validation.
        
        Input:
            target_variable: string
            problem: integer
            epochs: integer
            train_dataset: tensorflow.python.data.ops.dataset_ops.BatchDataset
            test_dataset: tensorflow.python.data.ops.dataset_ops.BatchDataset
            model: nn_model_refactor.NNModel
            results_file: string
            train_summary_writer: tensorflow.python.ops.summary_ops_v2.ResourceSummaryWriter
            test_summary_writer: tensorflow.python.ops.summary_ops_v2.ResourceSummaryWriter
            
        Return:
            None (void function)
            
        Example:
            early_stopping('abc', 0, 5, train_dataset, test_dataset, model, 'results.txt', train_summary_writer, test_summary_writer)
    """
        
    compile_params = compile(problem)
        
    if problem in [0, 1]:
        # Parameters which define convergetion for early stopping
        minimum_delta = 0.001
        patience = 3
            
        print('Early stopping parameters: patience = {}, minimum_delta = {}'.format(patience, minimum_delta),
                file=open(results_file, "a"))
            
        # Allocate parameters
        patience_val = np.zeros(patience)
        history = dict()
        history['accuracy'] = np.zeros(epochs)
        history['val_accuracy'] = np.zeros(epochs)
        history['loss'] = np.zeros(epochs)
        history['val_loss'] = np.zeros(epochs)
            
#             loss_object, optimizer, train_loss, train_accuracy, test_loss, test_accuracy = compile(problem)
            
    elif problem == 2:
        # To differentiate case when target variable is 'created'
        if target_variable == 'created':
            minimum_delta = 500
            patience = 3
        else:
            minimum_delta = 5
            patience = 3
                
        print('Early stopping parameters: patience = {}, minimum_delta = {}'.format(patience, minimum_delta),
                file=open(results_file, "a"))
        
        # Allocate parameters
        patience_val = np.zeros(patience)
        history = dict()
        history['r_square'] = np.zeros(epochs)
        history['val_r_square'] = np.zeros(epochs)
        history['loss'] = np.zeros(epochs)
        history['val_loss'] = np.zeros(epochs)

    # Iterate through epochs
    for epoch in range(epochs):
        print("Start of epoch {}".format(epoch + 1))
        start = time.time()
        
        # Iterate through train dataset
        for (x_train, y_train) in train_dataset:
            
            # Optimize the risk
            train_step(model, compile_params, x_train, y_train, problem, num_layers=3)
        
        # Write train parammeters in summary
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', compile_params['train_loss'].result(), step=epoch)
            if problem == 2:
                tf.summary.scalar('r_square', compile_params['train_metric'].result(), step=epoch)
            else:
                tf.summary.scalar('accuracy', compile_params['train_metric'].result(), step=epoch)

        # Iterate through test dataset
        for (x_test, y_test) in test_dataset:
            
            # Calculate the risk
            test_step(model, compile_params, x_test, y_test, problem, num_layers=3)
            
        # Write test parammeters in summary 
        with test_summary_writer.as_default():
            tf.summary.scalar('loss', compile_params['test_loss'].result(), step=epoch)
            if problem == 2:
                tf.summary.scalar('r_square', compile_params['train_metric'].result(), step=epoch)
            else:
                tf.summary.scalar('accuracy', compile_params['train_metric'].result(), step=epoch)

        # If problem is regression
        if problem == 2:
            template = 'Epoch {:d}, Loss: {:0.4f}, R Square: {:0.4f}, Validation Loss: {:0.4f}, Validation R Square: {:0.4f}'
            print(template.format(epoch + 1,
                                    compile_params['train_loss'].result(),
                                    compile_params['train_metric'].result().numpy() * 100,
                                    compile_params['test_loss'].result(),
                                    compile_params['test_metric'].result().numpy() * 100))
            print("Time taken for epoch {:d}: {:0.4f} secs\n".format(epoch + 1, time.time() - start))

            # Write parammeters in history 
            history['r_square'][epoch] = compile_params['train_metric'].result()
            history['loss'][epoch] = compile_params['train_loss'].result()
            history['val_loss'][epoch] = compile_params['test_loss'].result()
            history['val_r_square'][epoch] = compile_params['test_metric'].result()
            
        # If problem is classification 
        else:
            template = 'Epoch {:d}, Loss: {:0.4f}, Accuracy: {:0.4f}, Validation Loss: {:0.4f}, Validation Accuracy: {:0.4f}'
            print(template.format(epoch + 1,
                                    compile_params['train_loss'].result(),
                                    compile_params['train_metric'].result().numpy() * 100,
                                    compile_params['test_loss'].result(),
                                    compile_params['test_metric'].result().numpy() * 100))
            print("Time taken for epoch {:d}: {:0.4f} secs\n".format(epoch + 1, time.time() - start))

            # Write parammeters in history
            history['accuracy'][epoch] = compile_params['train_metric'].result()
            history['loss'][epoch] = compile_params['train_loss'].result()
            history['val_loss'][epoch] = compile_params['test_loss'].result()
            history['val_accuracy'][epoch] = compile_params['test_metric'].result()

        # If number of epochs is greater than 5
        if epoch > 4:
            # Check for gradient in consecutive iterations for validation loss (to check if convergence is reached)
            differences = np.abs(np.diff(history['val_loss'][(epoch - patience): epoch], n=1))
            check = differences > minimum_delta
            
            # If there are no or insufficient gradient or target variable is 'detailed_status' and epoch is 8
            if np.all(check == False) or (target_variable == 'detailed_status' and epoch == 7):
                # Write the results
                print('Optimizer config data: {}'.format(optimizer.get_config()), file=open(results_file, "a"))
                print('The number of classes:', n_classes, 'the number of training samples:', dataset_length, '.',
                        file=open(results_file, "a"))
                
                # Write parameters
                print(template.format(epoch + 1, 
                                        compile_params['train_loss'].result(),
                                        compile_params['train_metric'].result().numpy() * 100,
                                        compile_params['test_loss'].result(),
                                        compile_params['test_metric'].result().numpy() * 100), '.', file=open(results_file, "a"))
                    
                # Break the loop for iterating through test batches
                # Let's examine the next epoch if this is not the last one 
                break
        
        # Reset states for parameters before new epoch iteration
        compile_params['train_loss'].reset_states()
        compile_params['test_loss'].reset_states()
        compile_params['train_metric'].reset_states()
        compile_params['test_metric'].reset_states() 
    
                

def single_training(target_variable, problem, df, epochs, hashed_features, datetime_columns, numeric_columns, reference_date='2015-01-01', dict_path='./saved_dictionaries/', 
                    var_path='./saved_variables/', features_path='./saved_features/', log_path='logs/gradient_tape/', results_file="results.txt", 
                    buffer_size=100000, batch_size=2**11):
    """ Transforms original data, build neural network model, extracts all fearures and target variable. 
        Creates dataset for training and testing and runs training of neural network with early stopping.
        
        Input:
            target_variable: string
            problem: integer
            df: pd.DataFrame
            epochs: integer
            hashed_features: list of strings
            datetime_columns: list of strings
            numeric_columns: list of strings
            reference_date: string
            dict_path: string
            var_path: string
            features_path: string
            log_path: string
            results_file: string
            buffer_size: integer
            batch_size: integer
            
        Return:
            None (void function)
            
        Example:
            single_training('abc', 0, df, 5, hashed_features, datetime_columns, numeric_columns, '2015-01-01',                                                                                             './saved_dictionaries/', './saved_variables/', './saved_features/', 'logs/gradient_tape/', 
                            'results.txt', 100000, 2**11)
    """
    
    ddf = transform_dataframe(df, target_variable, problem, hashed_features, datetime_columns, numeric_columns, reference_date, dict_path)
    cardinalities, n_classes = dataset_measure(ddf, target_variable, problem, hashed_features, var_path)
    
    num_train_records, cat_train_records, target_train_records, num_test_records, cat_test_records, target_test_records = extract_features('./saved_features/', 'abc')
    
    model, inputs = build_nn_model(problem, cardinalities, n_classes, num_train_records.shape[1], cat_train_records.shape[1])
    
    # Form the train dataset
    train_dataset = tf.data.Dataset.from_tensor_slices(((cat_train_records, num_train_records), target_train_records))

    # Form the test dataset
    test_dataset = tf.data.Dataset.from_tensor_slices(((cat_test_records, num_test_records), target_test_records))
    
    # Reshuffle train dataset and define batch size
    train_dataset = train_dataset.shuffle(buffer_size, reshuffle_each_iteration=True).batch(batch_size)
    test_dataset = test_dataset.batch(batch_size)

    # Get current date and time
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Make train and test directories for logs
    train_log_dir = log_path + current_time + '/train'
    test_log_dir = log_path + current_time + '/test'
    
    # Create summary file writers for given log directories
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    print('\n\nTraining log for the target variable "{}" {}:'.format(target_variable, problem),
          file=open(results_file, "a"))
    
    early_stopping(target_variable, problem, epochs, train_dataset, test_dataset, model, results_file, train_summary_writer, test_summary_writer)
    