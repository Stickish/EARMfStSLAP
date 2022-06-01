from copyreg import pickle
from gc import callbacks
from json import load
from pickletools import optimize
from re import L
from subprocess import call
from tabnanny import verbose
from tkinter import E
import pydot
import pydotplus
import graphviz

import numpy as np
from sklearn import ensemble
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, Concatenate, Flatten, Conv2D, AveragePooling2D
from tensorflow.keras import Model, Input
from tensorflow.keras import optimizers
from tensorflow.keras.regularizers import l1, l2, l1_l2
from collections import defaultdict
from sklearn.decomposition import PCA
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping

class F_theta:
    """
        Create either a training or prediction model for creating target vectors to be used while extracting more abstract rules

        Input:
        ================================================================
        article_shape: The shape each article takes, shape=(maximum_number_of_words_in_a_article, number_of_unique_words_in_articles)
        train_predict: Wheter to create a training or prediction model, 0=training, 1=prediction
        embedding_dim: The final dimension of the network, the size of the target vectors
        separation: The separation for unsimilar products, see triple_loss
        verbose: Wheter or not to print the summary of the model
        weights_path: The path to save/load the weights of the model from 
        visualize: Wheter or not to save the model as a .png
        model_file_path: Where to save the visualized model
        """

    def __init__(self,
                 article_shape,
                 train_predict,
                 embedding_dim=64,
                 separation=10,
                 verbose=False,
                 weights_path='f_theta.h5',
                 visualize=False,
                 model_file_path=None,
                 optimizer='Adam'):
        assert optimizer in ['Adam', 'SGD']
        self.article_shape = article_shape
        self.embedding_dim = embedding_dim
        self.separation = separation
        self.verbose = verbose
        self.weights_path = weights_path
        self.visualize = visualize
        self.model_file_path = model_file_path
        self.optimizer = optimizer

        # Defining the layers
        self.dense_0 = Dense(self.embedding_dim*8,
                             name='dense_0',
                             kernel_regularizer=l2(0.01),
                             bias_regularizer=l2(0.01))

        self.dense_1 = Dense(self.embedding_dim*8,
                             name='dense_1',
                             kernel_regularizer=l2(0.01),
                             bias_regularizer=l2(0.01))

        self.dense_2 = Dense(self.embedding_dim*4,
                             name='dense_2',
                             kernel_regularizer=l2(0.01),
                             bias_regularizer=l2(0.01))

        self.dense_3 = Dense(self.embedding_dim,
                             name='dense_3',
                             kernel_regularizer=l2(0.01),
                             bias_regularizer=l2(0.01))
        
        self.dense_4 = Dense(self.embedding_dim,
                             name='dense_4',
                             kernel_regularizer=l2(0.01),
                             bias_regularizer=l2(0.01))

        self.dropout_01 = Dropout(0.1, name='dropout_01') # 0.1 chosen pretty arbitrarily
        self.dropout_11 = Dropout(0.1, name='dropout_11') # 0.1 chosen pretty arbitrarily
        self.dropout_21 = Dropout(0.1, name='dropout_21') # 0.1 chosen pretty arbitrarily
        self.dropout_31 = Dropout(0.1, name='dropout_31') # 0.1 chosen pretty arbitrarily

        self.dropout_02 = Dropout(0.1, name='dropout_02') # 0.1 chosen pretty arbitrarily
        self.dropout_12 = Dropout(0.1, name='dropout_12') # 0.1 chosen pretty arbitrarily
        self.dropout_22 = Dropout(0.1, name='dropout_22') # 0.1 chosen pretty arbitrarily
        self.dropout_32 = Dropout(0.1, name='dropout_32') # 0.1 chosen pretty arbitrarily

        self.leaky_relu_01 = LeakyReLU(alpha=0.3, name='leaky_relu_01')
        self.leaky_relu_11 = LeakyReLU(alpha=0.3, name='leaky_relu_11')
        self.leaky_relu_21 = LeakyReLU(alpha=0.3, name='leaky_relu_21')
        self.leaky_relu_31 = LeakyReLU(alpha=0.3, name='leaky_relu_31')

        self.leaky_relu_02 = LeakyReLU(alpha=0.3, name='leaky_relu_02')
        self.leaky_relu_12 = LeakyReLU(alpha=0.3, name='leaky_relu_12')
        self.leaky_relu_22 = LeakyReLU(alpha=0.3, name='leaky_relu_22')
        self.leaky_relu_32 = LeakyReLU(alpha=0.3, name='leaky_relu_32')

        if train_predict == 0:
            self.network = self.define_f_theta_train()

        elif train_predict == 1:
            self.network = self.define_f_theta_pred()


    def triple_loss(self):
        """
        Loss function for the network, groups similar articles (articles are considered similar if they occur in the same order)
        closer together and separates unsimilar articles

        Input:
        ====================================
        y_true: nothing, needed for the network to work
        y_pred: output from the network, shape = (batch_size, 3*embedding_dim)

        Output:
        ====================================
        losses: Tensor of losses with shape = (batch_size,)
        """

        def loss(y_true, y_pred):

            anchor = y_pred[:, :self.embedding_dim]
            pos = y_pred[:, self.embedding_dim:self.embedding_dim*2]
            neg = y_pred[:, self.embedding_dim*2:]

            pos_norm = tf.norm(anchor - pos, axis=1, keepdims=True)
            neg_norm = tf.norm(anchor - neg, axis=1, keepdims=True) ** self.separation

            losses = tf.math.divide(pos_norm, neg_norm)

            return tf.math.reduce_mean(losses)

        return loss


    def define_f_theta_train(self):
        """
        Defining and compiling the training model with triple loss

        Output:
        ================================================================
        f_theta: the compiled training model
        """      
        # Building the model
        inp_1 = Input(shape=self.article_shape, name='Anchor')
        inp_2 = Input(shape=self.article_shape, name='Positive')
        inp_3 = Input(shape=self.article_shape, name='Negative')

        # Branch 1 - Anchor
        x1 = self.dense_1(inp_1)
        x1 = self.leaky_relu_11(x1)
        x1 = self.dropout_11(x1)

        x1 = self.dense_2(x1)
        x1 = self.leaky_relu_12(x1)
        x1 = self.dropout_12(x1)

        x1 = self.dense_3(x1)
        x1 = self.dense_4(x1)

        # Branch 2 - Positive
        x2 = self.dense_1(inp_2)
        x2 = self.leaky_relu_21(x2)
        x2 = self.dropout_21(x2)

        x2 = self.dense_2(x2)
        x2 = self.leaky_relu_22(x2)
        x2 = self.dropout_22(x2)

        x2 = self.dense_3(x2)
        x2 = self.dense_4(x2)

        # Branch 3 - Negative
        x3 = self.dense_1(inp_3)
        x3 = self.leaky_relu_31(x3)
        x3 = self.dropout_31(x3)

        x3 = self.dense_2(x3)
        x3 = self.leaky_relu_32(x3)
        x3 = self.dropout_32(x3)

        x3 = self.dense_3(x3)
        x3 = self.dense_4(x3)

        concat = Concatenate()([x1, x2, x3])

        f_theta = Model(inputs=[inp_1, inp_2, inp_3], 
                        outputs=concat, 
                        name='f_train')

        # Using custom triple loss function
        if self.optimizer == 'Adam':
            f_theta.compile(optimizer=optimizers.Adam(learning_rate=0.0001), # Add self.lr here 
                            loss = self.triple_loss(),
                            run_eagerly=False)

        elif self.optimizer == 'SGD':
            f_theta.compile(optimizer=optimizers.SGD(learning_rate=0.0001), # Add self.lr here 
                            loss = self.triple_loss(),
                            run_eagerly=False)
        
        if self.verbose:
            print(f_theta.summary())

        if self.visualize:
            assert type(self.model_file_path) == str
            plot_model(f_theta, to_file=self.model_file_path)

        return f_theta
    
    
    def define_f_theta_pred(self):
        """
        Defining and compiling the prediction model from the trained weights of a training network

        Output:
        ================================================================
        f_theta_pred: a model for computing the target vectors for articles
        """        

        # Build the model
        inp_pred = Input(shape=self.article_shape, name='Pred input')
        
        x_p = self.dense_1(inp_pred)
        x_p = self.leaky_relu_11(x_p)
        x_p = self.dropout_11(x_p)

        x_p = self.dense_2(x_p)
        x_p = self.leaky_relu_12(x_p)
        x_p = self.dropout_12(x_p)

        x_p = self.dense_3(x_p)
        x_p = self.dense_4(x_p)

        f_theta_pred = Model(inputs=inp_pred, 
                             outputs=x_p, 
                             name='f_pred')

        # Load weights of f_theta_train
        f_theta_pred.load_weights(self.weights_path)

        if self.verbose:
            print(f_theta_pred.summary())

        if self.visualize:
            assert type(self.model_file_path) == str
            plot_model(f_theta_pred, to_file=self.model_file_path)

        return f_theta_pred
    

    def train(self, 
              X_train, y_train,
              val_data, num_workers=4,
              num_epochs=10, batch_size=64,
              early_stopping=False
              ):
        """
        Function for fitting the train model, is callable on the prediction network. However this shoudl never be done.

        Input:
        ================================================================
        X_train: Training data from articles in the same orders, shape = (n_article_combinations, 2, num_words_max, unique_words)
        y_train: Labels indicating similarity, 0 = unsimilar and 1 = similar with shape = (n_article_combinations, 1) TODO: Remove?
        validation_data: Tuple with validation data formatted like X_train and y_train, on the form ([X_val[:,0,:,:], X_val[:,1,:,:]], y_val)
        workers: Number of threads to use when fitting
        epochs: Number of epochs to use while fitting
        batch_size: How many articles to include in each batch

        Output:
        ================================================================
        Saves the weights of the network to weights_path
        """
        
        try:
           self.network.load_weights(self.weights_path)
        except:
           print('No bip bop in here yao')


        if early_stopping:

            callback = EarlyStopping(monitor='val_loss', 
                                     min_delta = 0.005, 
                                     mode='min', 
                                     patience=2, 
                                     restore_best_weights=True, 
                                     verbose=True)

            self.network.fit(x=[X_train[:,0,:], X_train[:,1,:], X_train[:,2,:]], 
                                    y=X_train[:,0,:],
                                    validation_data=val_data, # TODO: Change this to above for consistency
                                    workers=num_workers,
                                    epochs=num_epochs,
                                    shuffle=True,
                                    batch_size=batch_size,
                                    callbacks=[callback]
                                    )
        else:
            self.network.fit(x=[X_train[:,0,:], X_train[:,1,:], X_train[:,2,:]], 
                                y=X_train[:,0,:],
                                validation_data=val_data, # TODO: Change this to above for consistency
                                workers=num_workers,
                                epochs=num_epochs,
                                shuffle=True,
                                batch_size=batch_size
                                )
        
        self.network.save_weights(self.weights_path)

    def compute_embeddings(self, article_vectors, article_names,
                           map_load_path=None, map_save_path=None,
                           space_load_path=None, space_save_path=None):

        # Try to load the embedding map and embedding_space
        if type(map_load_path) == str and type(space_load_path) == str:
            try:
                # This has alot of ways to mix our stuff up real bad.
                map_file = open(map_load_path, 'rb')
                embedding_map = pickle.load(map_file)
                map_file.close()
                
                space_file = open(space_load_path, 'rb')
                embedding_space = pickle.load(space_file)
                space_file.close()

                return embedding_map, embedding_space
            
            except:
                print('No such file found')
        

        embedding_space = np.asarray(self.network(article_vectors)) # Want array not tensor
        embedding_map = {}

        for i in range(len(article_names)):
            embedding_map[article_names[i]] = embedding_space[i,:]

        # Save the embedding map
        if type(map_save_path) == str and type(space_save_path) == str:
            try:
                # Save the embedding map
                map_file = open(map_save_path, "wb")
                pickle.dump(embedding_map, map_file)
                map_file.close()

                space_file = open(space_save_path, "wb")
                pickle.dump(embedding_space, space_file)
                space_file.close()
                
                return embedding_map, embedding_space

            except:
                print('???')
        else:
            return embedding_map, embedding_space


        


