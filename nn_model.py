import tensorflow as tf

class NNModel(tf.keras.Model):
    """ This class defines the architecture of a neural network which concatenates 20 embedding
        layers for categorical inputs with numerical input of shape (33,1).
        Every categorical input is transformed to an embedded vector of the dimension
        dim = round[(cardinal number of the set of categorical variables)^0.25.
        After Embedding layer, we perform flattening in order to transform the vectors
        from the shape ( , 1, dim) to ( , dim). At the end there are 3 dense layers with
        dropout, batch normalization and ReLU non-linearity and one output layer
    """
    
    def __init__(self,
                 problem,
                 cardinalities,
                 n_classes,
                 embedding_dim_power=0.25,
                 dense=[256, 256, 128],
                 dropout=[0.1, 0.1, 0.1]):
        """ Constructor method.
            
            Input:
                self: nn_model_refactor.NNModel
                problem: integer
                cardinalities: list of integers
                n_classes: integer
                embedding_dim_power: float
                dense: list of integers
                dropout: list of floats
                
            Class attributes:
                problem: integer
                cardinalities: tensorflow.python.training.tracking.data_structures.ListWrapper
                n_classes: integer
                emb_pow: float
                n: tensorflow.python.training.tracking.data_structures.ListWrapper
                embedding: tensorflow.python.training.tracking.data_structures.ListWrapper
                flatten: tensorflow.python.keras.layers.core.Flatten
                concatenate: tensorflow.python.keras.layers.merge.Concatenate
                dense: tensorflow.python.training.tracking.data_structures.ListWrapper
                dropout: tensorflow.python.training.tracking.data_structures.ListWrapper
                batchnorm: tensorflow.python.training.tracking.data_structures.ListWrapper
                relu: tensorflow.python.keras.layers.advanced_activations.ReLU
                dense_classifier: tensorflow.python.keras.layers.core.Dense
                binary_classifier: tensorflow.python.keras.layers.core.Dense
                dense_neuron: tensorflow.python.keras.layers.core.Dense
        """

        super(NNModel, self).__init__()

        self.problem = problem
        self.cardinalities = cardinalities
        
        self.n_classes = n_classes

        self.emb_pow = embedding_dim_power
        
        self.n = cardinalities
        
        embedding_list = []
        for i in range(len(cardinalities)):
            embedding_list.append(tf.keras.layers.Embedding(self.n[i], int(self.n[i] ** self.emb_pow)))
        self.flatten = tf.keras.layers.Flatten()
        self.embedding = embedding_list

        self.concatenate = tf.keras.layers.Concatenate()
        
        dense_list = []
        dropout_list = []
        batchnorm_list = []
        for i in range(len(dense)):
            dense_list.append(tf.keras.layers.Dense(dense[i]))
            dropout_list.append(tf.keras.layers.Dropout(dropout[i]))
            batchnorm_list.append(tf.keras.layers.BatchNormalization())
        self.dense = dense_list
        self.dropout = dropout_list
        self.batchnorm = batchnorm_list
        self.relu = tf.keras.layers.ReLU()
        
        if self.problem == 0:
            self.dense_classifier = tf.keras.layers.Dense(n_classes, activation='softmax')
        elif problem == 1:
            self.binary_classifier = tf.keras.layers.Dense(1, activation='sigmoid')
        elif problem == 2:
            self.dense_neuron = tf.keras.layers.Dense(1, activation='linear')
    

            
    def call(self, inputs, num_layers):
        """ Method that calls neural network model.
        
            Input:
                self: nn_model_refactor.NNModel
                inputs: tuple
                num_layers: integer
                
            Return:
                x: tensorflow.python.framework.ops.EagerTensor
        """
        
        inputs_list = []
        categorical_features = tf.transpose(inputs[0])
        for i in range(categorical_features.shape[0]):
            x_temp = self.embedding[i](categorical_features[i])
            x_temp = self.flatten(x_temp)
            inputs_list.append(x_temp)
        inputs_list.append(inputs[1])

        x = self.concatenate(inputs_list)

        for i in range(num_layers):
            x = self.dense[i](x)
            x = self.dropout[i](x)
            x = self.batchnorm[i](x)
            x = self.relu(x)

        if self.problem == 0:

            x = self.dense_classifier(x)

        elif self.problem == 1:

            x = self.binary_classifier(x)

        elif self.problem == 2:

            x = self.dense_neuron(x)

        return x


    
def build_nn_model(ml_problem, cat_cardinalities, num_classes, num_len, cat_len, num_layers=3):
    """ Function that builds neural network model. 
        
        Input:
            ml_problem: integer
            cat_cardinalities: list of integers
            num_classes: integer
            num_len: integer
            cat_len: integer
            num_layers: integer
            
        Return:
            nn_model: nn_model_refactor.NNModel
            nn_inputs: list(tf.keras.layers.Input, tf.keras.layers.Input)
            
        Example:
            build_nn_model(ml_problem, cat_cardinalities, num_classes, num_layers=3)
    """

    numerical_inputs = tf.keras.layers.Input(shape=(num_len,))
    categorical_inputs = tf.keras.layers.Input(shape=(cat_len,))
    
    nn_inputs = [categorical_inputs, numerical_inputs]

    nn_model = NNModel(ml_problem, cat_cardinalities, num_classes)

    nn_model(nn_inputs, num_layers)

    print(nn_model.summary())

    return nn_model, nn_inputs
