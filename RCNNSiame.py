import sys
import numpy as np
# import pandas as pd
from scipy.misc import imread,imsave
from skimage.transform import rescale
import pickle
import os
import matplotlib.pyplot as plt

plt.switch_backend('agg')

import cv2
import time

import tensorflow as tf
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate,ConvLSTM2D, TimeDistributed
from keras.models import Model

from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform

from keras.engine.topology import Layer
from keras.regularizers import l2
from keras import backend as K

from sklearn.utils import shuffle

import numpy.random as rng





train_folder = "../Training_data/"
val_folder = '../Validation_data/'
save_path = '../data/'


def get_siamese_model(input_shape):

    
    # Define the tensors for the two input images
    left_input = Input(input_shape)
    right_input = Input(input_shape)
    
    # Convolutional Neural Network [Edit here to use different models] 
    model = Sequential()
    model.add(ConvLSTM2D(16, (3, 3),kernel_initializer=initialize_weights, activation='relu',padding='same', return_sequences=True,input_shape=input_shape))
    model.add(TimeDistributed(MaxPooling2D((4,4))))
    model.add(ConvLSTM2D(16, (3, 3),kernel_initializer=initialize_weights, activation='relu',padding='same', return_sequences=True,input_shape=input_shape))
    model.add(TimeDistributed(MaxPooling2D((4,4))))
    model.add(ConvLSTM2D(16, (3, 3),kernel_initializer=initialize_weights, activation='relu',padding='same', return_sequences=True,input_shape=input_shape))
    model.add(TimeDistributed(MaxPooling2D((4,4))))

    model.add(Flatten())
    model.add(Dense(2048, activation='sigmoid',
                   kernel_regularizer=l2(1e-3),
                   kernel_initializer=initialize_weights,
                     bias_initializer=initialize_bias,))
    
    # Generate the encodings (feature vectors) for the two images
    encoded_l = model(left_input)
    encoded_r = model(right_input)
    
    # Add a customized layer to compute the absolute difference between the encodings
    L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])
    
    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = Dense(1,activation='sigmoid')(L1_distance)
    
    # Connect the inputs with the outputs
    siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)
    
    # return the model
    return siamese_net


def initialize_weights(shape, name=None):
    """
        The paper, http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
        suggests to initialize CNN layer weights with mean as 0.0 and standard deviation of 0.01
    """
    return np.random.normal(loc = 0.0, scale = 1e-2, size = shape)

def initialize_bias(shape, name=None):
    """
        The paper, http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
        suggests to initialize CNN layer bias with mean as 0.5 and standard deviation of 0.01
    """
    return np.random.normal(loc = 0.5, scale = 1e-2, size = shape)

def loadimgs(path,n = 0):
    '''
    path => Path of train directory or test directory
    '''
    X=[]
    y = []
    cat_dict = {}
    Angle = {}
    curr_y = n


    for Individual in os.listdir(path):
        print("loading Individual: " + Individual)
        Angle[Individual] = [curr_y,None]
        Individual_path = os.path.join(path,Individual)
        

        for angle in os.listdir(Individual_path):
            cat_dict[curr_y] = (Individual, angle)
            category_images=[]
            angle_path = os.path.join(Individual_path, angle)
            

            for filename in os.listdir(angle_path):
                image_path = os.path.join(angle_path, filename)
                image = imread(image_path)
                img_res = rescale(image, 1.0 / 10.0 )
                image = img_res #.astype(int)
                category_images.append(image)
                y.append(curr_y)
            try:
                X.append(np.stack(category_images))

            except ValueError as e:
                print(e)
                print("error - category_images:", category_images)
            curr_y += 1
            Angle[Individual][1] = curr_y - 1

    y = np.vstack(y)
    X = np.stack(X)
    return X,y,lang_dict


# Load the images and save them in a pickle file, comment out to keep using the pickle file data.

X,y,c = loadimgs(train_folder)

print (X.shape, y.shape, c.keys())

with open(os.path.join(save_path,"train.pickle"), "wb") as f:
    pickle.dump((X,c),f)

Xval,yval,cval=loadimgs(val_folder)

print (Xval.shape, yval.shape, cval.keys())


with open(os.path.join(save_path,"val.pickle"), "wb") as f:
    pickle.dump((Xval,cval),f)




def get_batch(batch_size,s="train"):
    """
    Create batch of n pairs, half same class, half different class

    """
    if s == 'train':
        X = Xtrain
        categories = train_classes
    else:
        X = Xval
        categories = val_classes
    n_classes, n_examples, w, h = X.shape #n classes is my individuals and examples pics of ind
    
    # randomly sample several classes to use in the batch
    categories = shuffle([ x for x in range(0,n_classes,2)])[:batch_size]
    categories_2 = shuffle([ x for x in range(1,n_classes,2)])

    # initialize 2 empty arrays for the input image batch
    pairs=[np.zeros((batch_size,n_examples, w, h,1)) for i in range(2)]
    
    # initialize vector for the targets
    targets=np.zeros((batch_size,))
    
    # make one half of it '1's, so 2nd half of batch has same class
    targets[batch_size//2:] = 1
    for i in range(batch_size):
        category = categories[i]

        # pick a category index, find this in X and enter it in pairs[0]
        pairs[0][i,:,:,:,:] = X[category].reshape(n_examples,w, h, 1) 

        if i >= batch_size // 2:
            # Load the same category as before but at a different angle
            category_2 = category  + 1

        else: 
            # load a different category as before also at a different angle
            new_cat_2 = []

            for j in range(len(categories_2)):
                if categories_2[j] == category + 1:
                    pass
                else:
                    new_cat_2.append(categories_2[j])

            category_2 = shuffle(new_cat_2)[0]

        pairs[1][i,:,:,:,:] = X[category_2].reshape(n_examples,w,h,1)

    return pairs, targets


def generate(batch_size, s="train"):
    """ a generator for batches, so model.fit_generator can be used. """
    while True:
        pairs, targets = get_batch(batch_size,s)
        yield (pairs, targets)


# Initialise the model and return a summary
model = get_siamese_model(( 20,96,128, 1))
model.summary()



optimizer = Adam(lr = 0.00006)
model.compile(loss="binary_crossentropy",optimizer=optimizer)

# open the pickle files with the data.
with open(os.path.join(save_path, "train.pickle"), "rb") as f:
    (Xtrain, train_classes) = pickle.load(f)
    
print("Training Individuals: \n")
print(list(train_classes.keys()))
pc = list(train_classes.keys())


with open(os.path.join(save_path, "val.pickle"), "rb") as f:
    (Xval, val_classes) = pickle.load(f)

print("Validation Individuals:")
print(list(val_classes.keys()))
pcval =list(val_classes.keys())

def make_oneshot_task(N, s="val", language=None):
    """Create pairs of test image, support set for testing N way one-shot learning. """
    if s == 'train':
        X = Xtrain
        categories = train_classes
    else:
        X = Xval
        categories = val_classes
    n_classes, n_examples, w, h = X.shape
    # true individual
    categories = shuffle([ x for x in range(0,n_classes,2)])

    # different categories
    categories_2 = shuffle([ x for x in range(1,n_classes,2)])

    # the first individual is the true one
    true_category = categories[0]

    # the same individual but different angle
    true_opp_angle_cat = true_category + 1

    # make sure when we are loading a different individual
    new_cat_2 = []
    for j in range(len(categories_2)):
                if categories_2[j] == true_opp_angle_cat:
                    pass
                else:
                    new_cat_2.append(categories_2[j])
    # finalise the pair. at [0] we have the other angle for the same individual  
    new_cat_2.insert(0,true_opp_angle_cat)

    
    # now take the true category at the first angle and create a vector spanning all test pairs
    test_image = np.asarray([X[true_category]]*N).reshape(N,n_examples, w, h,1)

    
    new_cat_2 = new_cat_2[:N]
    support_set = X[new_cat_2,:]
    support_set = support_set.reshape(N,n_examples, w, h,1)

    targets = np.zeros((N,))
    targets[0] = 1
    targets, test_image, support_set = shuffle(targets, test_image, support_set)

    # load the pairs for the test
    pairs = [test_image,support_set]


    return pairs, targets




def test_oneshot(model, N, k, s = "val", verbose = 0):
    """Test average N way oneshot learning accuracy of a siamese neural net over k one-shot tasks"""
    n_correct = 0
    if verbose:
        print("Evaluating model on {} random {} way one-shot learning tasks ... \n".format(k,N))
    for i in range(k):
        inputs, targets = make_oneshot_task(N,s)
        probs = model.predict(inputs)
        if np.argmax(probs) == np.argmax(targets):
            n_correct+=1
    percent_correct = (100.0 * n_correct / k)
    if verbose:
        print("Got an average of {}% {} way one-shot learning accuracy \n".format(percent_correct,N))
    return percent_correct


# Hyper parameters
evaluate_every = 10000# interval for evaluating on one-shot tasks 200
batch_size = 8
n_iter = 20000 # No. of training iterations 200
N_way = 70 # how many classes for testing one-shot tasks
n_val = 50 # how many one-shot tasks to validate on 250
best = -1

# where the model will be saved
model_path = '../weights/'

print("Starting training process!")
print("-------------------------------------")
t_start = time.time()
for i in range(1, n_iter+1):
    (inputs,targets) = get_batch(batch_size)
    #print ('input presetned to model for training',inputs[1].shape) #input [ [10 images from one class]  [ 10 images form another class]]
    loss = model.train_on_batch(inputs, targets)
    if i % evaluate_every == 0:
        print("\n ------------- \n")
        print("Time for {0} iterations: {1} mins".format(i, (time.time()-t_start)/60.0))
        print("Train Loss: {0}".format(loss)) 
        val_acc = test_oneshot(model, N_way, n_val, verbose=True)
        print (val_acc)
        model.save_weights(os.path.join(model_path, 'weights.{}.h5'.format(i)))
        if val_acc >= best:
            print("Current best: {0}, previous best: {1}".format(val_acc, best))
            best = val_acc




weights_iteration = n_iter

model.load_weights(os.path.join(model_path, "weights." + str(weights_iteration) +".h5"))

def nearest_neighbour_correct(pairs,targets):
    """returns 1 if nearest neighbour gets the correct answer for a one-shot task
        given by (pairs, targets)"""
    L2_distances = np.zeros_like(targets)
    for i in range(len(targets)):
        L2_distances[i] = np.sum(np.sqrt(pairs[0][i]**2 - pairs[1][i]**2))
    if np.argmin(L2_distances) == np.argmax(targets):
        return 1
    return 0


def test_nn_accuracy(N_ways,n_trials):
    """Returns accuracy of NN approach """
    print("Evaluating nearest neighbour on {} unique {} way one-shot learning tasks ...".format(n_trials,N_ways))

    n_right = 0
    
    for i in range(n_trials):
        pairs,targets = make_oneshot_task(N_ways,"val")
        correct = nearest_neighbour_correct(pairs,targets)
        n_right += correct
    return 100.0 * n_right / n_trials



ways = np.arange(1,70,5)
resume =  False
trials = 50


val_accs, train_accs,nn_accs = [], [], []
for N in ways: 
    val_accs.append(test_oneshot(model, N, trials, "val", verbose=True))
    train_accs.append(test_oneshot(model, N, trials, "train", verbose=True))
    print("---------------------------------------------------------------------------------------------------------------")


with open(os.path.join(save_path,"accuracies.pickle"), "wb") as f:
    pickle.dump((val_accs,train_accs,nn_accs),f)

with open(os.path.join(save_path, "accuracies.pickle"), "rb") as f:
    (val_accs, train_accs, nn_accs) = pickle.load(f)





# 
# Save the accuracies on disk
with open(os.path.join(save_path,"accuracies.pickle"), "wb") as f:
    pickle.dump((val_accs,train_accs,nn_accs),f)


# Load the accuracies from disk
with open(os.path.join(save_path, "accuracies.pickle"), "rb") as f:
    (val_accs, train_accs, nn_accs) = pickle.load(f)


fig,ax = plt.subplots(1)
ax.plot(ways, val_accs, "r", label="RCNNS val")
ax.plot(ways, train_accs, "m--", label="RCNNS train")
print ('WAYS',ways)
ax.plot(ways, 100.0/ways, "g", label="Random")
plt.xlabel("Number of classes compared to the sample")
plt.ylabel("Accuracy")
plt.title("OU-MVLP RCNN Siamese Network")
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()
plt.savefig('Output.png')

