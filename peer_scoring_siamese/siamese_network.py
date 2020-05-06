import numpy as np
import tensorflow.keras as keras
import pandas as pd

class SiameseNN:
    @staticmethod
    def _triplet_loss(y_true, y_pred):
        """
        Implementation of the triplet loss function
        Arguments:
        y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
        y_pred -- python list containing three objects:
                anchor -- the encodings for the anchor data
                positive -- the encodings for the positive data (similar to anchor)
                negative -- the encodings for the negative data (different from anchor)
        Returns:
        loss -- real number, value of the loss

        $$L=\max(d(A,P)-d(A,N)+\alpha,0)$$
        """
        total_lenght = y_pred.shape[-1]

        anchor = y_pred[:,0:int(total_lenght/3)]
        positive = y_pred[:,int(total_lenght/3):int(total_lenght*2/3)]
        negative = y_pred[:,int((total_lenght*2)/3):int(total_lenght)]

        # distance between the anchor and the positive
        pos_dist = keras.backend.sum(keras.backend.square(anchor-positive),axis=1)

        # distance between the anchor and the negative
        neg_dist = keras.backend.sum(keras.backend.square(anchor-negative),axis=1)

        # compute loss
        #return keras.backend.mean(keras.backend.maximum(pos_dist-neg_dist+0.1,0.0))
        #print(depp.shape)
        return keras.backend.maximum(pos_dist-neg_dist+0.1,0.0)

    @staticmethod
    def _triplet_loss_alpha(y_true, y_pred):
        """
        Implementation of the triplet loss function where alpha is taken from y_true
        Arguments:
        y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
        y_pred -- python list containing three objects:
                anchor -- the encodings for the anchor data
                positive -- the encodings for the positive data (similar to anchor)
                negative -- the encodings for the negative data (different from anchor)
        Returns:
        loss -- real number, value of the loss

        $$L=\max(d(A,P)-d(A,N)+\alpha,0)$$
        """
        total_lenght = y_pred.shape[-1]

        anchor = y_pred[:,0:int(total_lenght/3)]
        positive = y_pred[:,int(total_lenght/3):int(total_lenght*2/3)]
        negative = y_pred[:,int((total_lenght*2)/3):int(total_lenght)]

        # distance between the anchor and the positive
        pos_dist = keras.backend.sum(keras.backend.square(anchor-positive),axis=1)

        # distance between the anchor and the negative
        neg_dist = keras.backend.sum(keras.backend.square(anchor-negative),axis=1)

        # compute loss
        #return keras.backend.mean(keras.backend.maximum(pos_dist-neg_dist+0.1,0.0))
        #print(keras.backend.mean(depp.shape))
        #print(keras.backend.mean(y_pred[:,3]))
        return keras.backend.maximum(pos_dist-neg_dist+0.5*y_true[:,3],0.0)

    @staticmethod
    def _triplet_loss_alpha_target(y_true, y_pred):
        """
        Implementation of the triplet loss function
        Arguments:
        y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
        y_pred -- python list containing three objects:
                anchor -- the encodings for the anchor data
                positive -- the encodings for the positive data (similar to anchor)
                negative -- the encodings for the negative data (different from anchor)
        Returns:
        loss -- real number, value of the loss

        $$L=\max(d(A,P)-d(A,N)+\alpha,0)$$
        """
        total_lenght = y_pred.shape[-1]

        anchor = y_pred[:,0:int(total_lenght/3)]
        positive = y_pred[:,int(total_lenght/3):int(total_lenght*2/3)]
        negative = y_pred[:,int((total_lenght*2)/3):int(total_lenght)]

        # distance between the anchor and the positive
        pos_dist = keras.backend.sum(keras.backend.square(anchor-positive),axis=1)

        # distance between the anchor and the negative
        neg_dist = keras.backend.sum(keras.backend.square(anchor-negative),axis=1)

        # compute loss
        #return keras.backend.mean(keras.backend.maximum(pos_dist-neg_dist+0.1,0.0))
        #print(keras.backend.mean(depp.shape))
        #print(keras.backend.mean(y_pred[:,3]))
        return 3.0*keras.backend.maximum(pos_dist-neg_dist+3.0*y_true[:,3],0.0) + keras.backend.maximum(pos_dist-1.1*y_true[:,0],0.0) + keras.backend.maximum(0.9*y_true[:,0]-pos_dist,0.0)

    @staticmethod 
    def save(model, file):
        keras.models.save(file)

    @staticmethod
    def load(file, newest_subdir = None):
        #if newest_subdir is None:
        return keras.models.load_model(file, custom_objects={'_triplet_loss': SiameseNN._triplet_loss, 
        '_triplet_loss_alpha': SiameseNN._triplet_loss_alpha, '_triplet_loss_alpha_target': SiameseNN._triplet_loss_alpha_target})

    @staticmethod
    def get_submodel(m):
        return m.get_layer(name='distance_model')
        
   
    @staticmethod
    def _create_simple_network(n_neurons, activation='relu',
                        kernel_regularizer=None, bias_regularizer=None, 
                        input_dim=1, 
                        normalize_output=False):
        keras.backend.clear_session()
        np.random.seed(42)
        model = keras.models.Sequential(name='distance_model')
        model.add(keras.layers.Dense(n_neurons[0], activation=activation, input_dim=input_dim, kernel_regularizer=kernel_regularizer, 
                        bias_regularizer=bias_regularizer, name='distance_model_layer_0')) 
        for i,n in enumerate(n_neurons[1:]):
            name='distance_model_layer_'+str(i+1)
            model.add(keras.layers.Dense(n, activation=activation, kernel_regularizer=kernel_regularizer, 
                        bias_regularizer=bias_regularizer,  name=name)) 
        if normalize_output:
            model.add(keras.layers.Lambda(lambda t: keras.backend.l2_normalize(t, axis=1)))
        return model

    @staticmethod
    def create(n_neurons, activation='relu', kernel_regularizer=None,
            bias_regularizer=None, input_dim=1, optimizer = None, 
            normalize_output = False):
        """If optimizer is not None, it also directly compiles the model
        """
        model = SiameseNN._create_simple_network(n_neurons, activation=activation, 
                kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, 
                input_dim=input_dim, normalize_output=normalize_output)
        input_dim = model.layers[0].input_shape[1]
        # Define the tensors for the three input images
        anchor_input = keras.layers.Input((input_dim, ), name="anchor_input")
        positive_input = keras.layers.Input((input_dim, ), name="positive_input")
        negative_input = keras.layers.Input((input_dim, ), name="negative_input") 
        
        # Generate the encodings (feature vectors) for the three images
        encoded_a = model(anchor_input)
        encoded_p = model(positive_input)
        encoded_n = model(negative_input)
        
        merged_vector = keras.layers.concatenate([encoded_a, encoded_p, encoded_n], axis=-1, name='merged_layer')
        #merged_vector = keras.backend.stack([encoded_a, encoded_p, encoded_n])
        # Connect the inputs with the outputs
        siamese = keras.models.Model(inputs=[anchor_input,positive_input,negative_input],outputs=merged_vector, name='siamese')
        if optimizer is not None:
            siamese.compile(loss=SiameseNN._triplet_loss, optimizer=optimizer)
        return siamese

def get_data(data, vols, data_dir):
    positive = None
    negative = None
    anchor = None
    info = None
    distance = None

    for vol in vols:
        for d in data:
            file_prefix = data_dir+d+'_' + str(vol)
            if positive is None:
                positive=pd.read_csv(file_prefix+'_positive.csv')#, index_col = 0))
                negative=pd.read_csv(file_prefix+'_negative.csv')#, index_col = 0))
                anchor=pd.read_csv(file_prefix+'_anchor.csv')#, index_col = 0))
                info=pd.read_csv(file_prefix+'_info.csv')#, index_col = 0))
                distance=pd.read_csv(file_prefix+'_distance.csv')#, index_col = 0))
            else:
                positive = positive.append(pd.read_csv(file_prefix+'_positive.csv'), ignore_index = True)#, index_col = 0))
                negative = negative.append(pd.read_csv(file_prefix+'_negative.csv'), ignore_index = True)#, index_col = 0))
                anchor = anchor.append(pd.read_csv(file_prefix+'_anchor.csv'), ignore_index = True)#, index_col = 0))
                info = info.append(pd.read_csv(file_prefix+'_info.csv'), ignore_index = True)#, index_col = 0))
                distance=distance.append(pd.read_csv(file_prefix+'_distance.csv'), ignore_index = True)#, index_col = 0))
    return anchor, positive, negative, info, distance

def get_training_data(data, vols, data_dir, shuffle = True, frac=1):
    anchor, positive, negative, info, distance = get_data(data, vols , data_dir)
    #now shuffle data
    if shuffle:
        anchor = anchor.sample(frac=frac)
        positive = positive.loc[anchor.index]
        negative = negative.loc[anchor.index]
        info = info.loc[anchor.index]
        distance = distance.loc[anchor.index]
    x_train=[anchor.values, positive.values, negative.values]
    y_train = distance.values
    return x_train, y_train, info