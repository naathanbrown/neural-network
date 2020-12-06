import numpy as np
import numpy.matlib 
import math
import matplotlib.pyplot as plt
import csv
import scipy.io as spio
from scipy import stats

'''
Calculates the softmax probility of the output layer
PARAMS
output - Array to be returned in softmax form
RETURNS
Array in softmax format
'''
def softmax(output):   
    c = np.max(output) 
    return np.exp(output-c)/np.sum(np.exp(output-c))

'''
Will divide every element passed to it by 255
PARAMS
row - This is a row of input, this case 3024
RETURNS
A row that has been divded each element by 255
'''
def normalise(row):
    return row / 255

'''
Plots the error on a figure then shows it
'''
def plot_err(errors, validation):
    #PLOT THE ERROR
    #plt.plot(errors)
    plt.plot(validation)
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.title('Average error per epoch - 1.0 dropout, ReLU, eta 0.075')
    plt.show()

'''
Performs all training and then eventually prints and accuracy score of test set once all epochs have finished
PARAMS
epochs - How many epochs the system will complete
batches - How many batches in the dataset
layer1 - How many nodes in layer1
layer2 - How many nodes in layer2
sigmod - True use sigmoid, false use ReLU
eta - Learning rate
drop - The dropout chance, 0.8 means each node has a 20% to be dropped
augment - Augment the data, true or false
noise_bool - Do you want to add noise to a random 1000 images
RETURNS
Array of error values
'''  
def train_network(epochs, batches, layer1, layer2, sigmoid, eta, drop, augment, noise_bool):

    augment = augment

    #load in the data
    mat = spio.loadmat('train_data.mat', squeeze_me=True)
    x_train = mat['x_train'] # data
    trainlabels_all = mat['x_train_labs'] # labels

    re_list_hoz = []
    re_list_ver = []

    #Augument the data if required, will up data to 150K samples
    if augment:
        for im in x_train:
            #perform a vertical flip
            im_r = im[0:1024].reshape(32, 32)
            im_g = im[1024:2048].reshape(32, 32)
            im_b = im[2048:].reshape(32, 32)
            im_r = np.flipud(im_r)
            im_g = np.flipud(im_g)
            im_b = np.flipud(im_b)
            re = np.concatenate((im_r.reshape(1024,), im_g.reshape(1024,)), axis=0)
            re = np.concatenate((re, im_b.reshape(1024,)), axis=0)

            re_list_ver.append(re)

            #perform a horizontal flip
            im_r = im[0:1024].reshape(32, 32)
            im_g = im[1024:2048].reshape(32, 32)
            im_b = im[2048:].reshape(32, 32)
            im_r = np.fliplr(im_r)
            im_g = np.fliplr(im_g)
            im_b = np.fliplr(im_b)
            re = np.concatenate((im_r.reshape(1024,), im_g.reshape(1024,)), axis=0)
            re = np.concatenate((re, im_b.reshape(1024,)), axis=0)

            re_list_hoz.append(re)

        x_train = np.concatenate((x_train, re_list_ver))
        x_train = np.concatenate((x_train, re_list_hoz))
        trainlabels = np.concatenate((np.concatenate((trainlabels,trainlabels)), trainlabels))

    mat2 = spio.loadmat('test_data.mat', squeeze_me=True)
    x_test = mat2['x_test'] # data
    testlabels = mat2['x_test_labs'] # labels

    #normalise the data
    x_test = np.apply_along_axis(normalise,1,x_test)
    x_train_all = np.apply_along_axis(normalise,1,x_train)

    shuffled_idxs = np.random.permutation(x_train_all.shape[0])
     
    x_valid = np.zeros([10000,3072])
    x_train = np.zeros([40000,3072])
    valid_labels = np.zeros([10000])
    trainlabels = np.zeros([40000])

    valid_size = x_valid.shape[0]

    for x in range(len(shuffled_idxs)):
        idx = shuffled_idxs[x]
        if x < valid_size:
            x_valid[x] = x_train_all[idx]
            valid_labels[x] = trainlabels_all[idx]
        else:
            x_train[x-valid_size] = x_train_all[idx]
            trainlabels[x-valid_size] = trainlabels_all[idx]

    #create sample length and imagesize
    n_samples, img_size = x_train.shape

    #number of output labels, 10 img classes
    nlabels = 10

    #turn lables into output vectors
    y_train = np.zeros((trainlabels.shape[0], nlabels))
    y_test  = np.zeros((testlabels.shape[0], nlabels))
    y_valid  = np.zeros((valid_labels.shape[0], nlabels))

    #Minus one is because no zero index, so class 1 is 0 in array
    for i in range(0,trainlabels.shape[0]):   
        y_train[i, (trainlabels[i].astype(int)-1)]=1   
    for i in range(0,testlabels.shape[0]):    
        y_test[i, (testlabels[i].astype(int)-1)]=1
    for i in range(0,valid_labels.shape[0]):    
        y_valid[i, (valid_labels[i].astype(int)-1)]=1

    #Hyperprams
    n_epoch = epochs
    n_batches = batches
    batch_size = math.ceil(n_samples/n_batches)

    n_input_layer  = img_size
    n_hidden_layer = layer1
    n_output_layer = nlabels

    sigmoid = sigmoid

    # Add another hidden layer
    n_hidden_layer2 = layer2 # number of neurons of the hidden layer. 0 deletes this layer

    #learning rate
    eta = eta

    #Threshold for if to drop a node, 1.0 is 0% change, 0.8 is 20% chance etc.
    drop = drop

    if sigmoid:
        #Do a Xavier Init
        W1 = np.random.randn(n_hidden_layer, n_input_layer) * np.sqrt(1 / (n_input_layer))
        if n_hidden_layer2>0:
            W2 = np.random.randn(n_hidden_layer2, n_hidden_layer) * np.sqrt(1 / (n_hidden_layer))
            W3 = np.random.randn(n_output_layer, n_hidden_layer2) * np.sqrt(1 / (n_hidden_layer2))
        else:
            W2 = np.random.randn(n_output_layer, n_hidden_layer) * np.sqrt(1 / (n_hidden_layer))
    else:
        #He-et-al initilisation for ReLU
        W1 = np.random.randn(n_hidden_layer, n_input_layer) * np.sqrt(2 / (n_input_layer + n_hidden_layer))
        if n_hidden_layer2>0:
            W2 = np.random.randn(n_hidden_layer2, n_hidden_layer) * np.sqrt(2 / (n_hidden_layer + n_hidden_layer2))
            W3 = np.random.randn(n_output_layer, n_hidden_layer2) * np.sqrt(2 / (n_hidden_layer2 + n_output_layer))
        else:
            W2 = np.random.randn(n_output_layer, n_hidden_layer) * np.sqrt(2 / (n_hidden_layer + n_output_layer))

    #Initiliase the bias
    bias_W1 = np.ones((n_hidden_layer,))*(np.mean(-x_train))
    bias_W2 = np.ones((n_output_layer,))*(-0.5)

    if n_hidden_layer2>0:    
        bias_W3=np.ones((n_output_layer,))*(-0.5)
        bias_W2=np.ones((n_hidden_layer2,))*(-0.5)

    errors=np.zeros((n_epoch,))
    valid=np.zeros((n_epoch,))
    accuracy_all=np.zeros((n_epoch,))

    print(n_hidden_layer2)

    for i in range(0,n_epoch):
        
        #Shuffle the order of the samples each epoch
        shuffled_idxs = np.random.permutation(n_samples)
        
        for batch in range(0,n_batches):
            # Initialise the gradients for each batch
            dW1 = np.zeros(W1.shape)
            dW2 = np.zeros(W2.shape)

            dbias_W1 = np.zeros(bias_W1.shape)
            dbias_W2 = np.zeros(bias_W2.shape)
            
            if n_hidden_layer2 > 0:
                dW3 = np.zeros(W3.shape)
                dbias_W3 = np.zeros(bias_W3.shape)

            # Loop over all the samples in the batch
            for j in range(0,batch_size):

                # Input (random element from the dataset)
                idx = shuffled_idxs[batch*batch_size + j]
                x = x_train[idx]

                #Add guassian noise to random images, 1000 each epoch 
                if j % 20 == 0 and noise_bool: 
                    noise = np.random.normal(0,0.02,3072)
                    x = x + noise

                # Form the desired output, the correct neuron should have 1 the rest 0
                desired_output = y_train[idx]

                # Neural activation: input layer -> hidden layer
                act1 = np.dot(W1,x)+bias_W1

                #Perform the random node drop for input to hidden layer
                if drop < 1.0:
                    drop1 = np.random.rand(act1.shape[0])
                    drop1 = drop1<drop
                    act1 = np.multiply(drop1,act1)
                    act1 = act1/drop

                if sigmoid:
                    # Apply the sigmoid function
                    out1 = 1/(1+np.exp(-act1))
                else:
                    #apply ReLU
                    out1 = np.clip(act1,0.0000001,None)

                # Neural activation: hidden layer -> output layer
                act2 = np.dot(W2,out1)+bias_W2

                #Perform the random node drop for hidden to output layer
                if drop < 1.0:
                    drop2 = np.random.rand(act2.shape[0])
                    drop2 = drop2<drop
                    act2 = np.multiply(drop2,act2)
                    act2 = act2/drop

                #Only run if 2nd layer
                if n_hidden_layer2 > 0:
                    if sigmoid:
                        out2 = 1/(1+np.exp(-act2))
                    else:
                        out2 = np.clip(act2,0.0000001,None)
                    # Neural activation: hidden layer 1 -> hidden layer 2
                    act3 = np.dot(W3,out2)+bias_W3
                    
                    #Perform the random node drop
                    if drop < 1.0:
                        drop3 = np.random.rand(act3.shape[0])
                        drop3 = drop3<drop
                        act3 = np.multiply(drop3,act3)
                        act3 = act3/drop

                    # Apply the sigmoid function
                    out3 = act3
                    
                    # Compute the error signal
                    e_n = softmax(out3) - desired_output
                    
                    # Backpropagation: output layer -> hidden layer 2
                    out3delta = e_n

                    #Perform the random node drop for back prop
                    if drop < 1.0:
                        out3delta = np.multiply(out3delta, drop3)
                        out3delta = out3delta/drop
                    
                    dW3 += np.outer(out3delta,out2)
                    dbias_W3 += out3delta
                    
                    # Backpropagation: hidden layer -> input layer
                    if sigmoid:
                        out2delta = out2 * (1-out2) *  np.dot(W3.T, out3delta)
                    else:
                        #set all values to 1 or 0 for gradient
                        out2[out2 > 0] = 1
                        out2[out2 <= 0] = 0
                        out2delta = out2 * np.dot(W3.T, out3delta)

                else:
                    # Compute the error signal
                    #No function as final layer
                    out2 = act2

                    #Error deriv
                    e_n = softmax(out2) - desired_output
                    
                    # Backpropagation: output layer -> hidden layer
                    #local grad, deviv of backprop
                    out2delta = e_n
                    
                if drop < 1.0:
                    out2delta = np.multiply(out2delta, drop2)
                    out2delta = out2delta/drop

                dW2 += np.outer(out2delta, out1)
                dbias_W2 += out2delta

                # Backpropagation: hidden layer -> input layer
                if sigmoid:
                    out1delta = out1 * (1-out1) * np.dot(W2.T, out2delta)
                else:
                    out1[out1 >= 0] = 1
                    out1[out1 < 0] = 0
                    out1delta = out1 *  np.dot(W2.T, out2delta)
                    
                if drop < 1.0:
                    out1delta = np.multiply(out1delta, drop1)
                    out1delta = out1delta/drop

                dW1 += np.outer(out1delta,x)
                dbias_W1 += out1delta

                # Store the error per epoch
                #total error
                #Full entropy loss
                if n_hidden_layer2 > 0:
                    errors[i] = errors[i] - np.sum(desired_output * np.log(softmax(out3)))/n_samples
                else:
                    errors[i] = errors[i] - np.sum(desired_output * np.log(softmax(out2)))/n_samples


            # After each batch update the weights using accumulated gradients
            W2 += -eta*dW2/batch_size
            W1 += -eta*dW1/batch_size

            bias_W1 += -eta*dbias_W1/batch_size
            bias_W2 += -eta*dbias_W2/batch_size
            
            if n_hidden_layer2 > 0:
                W3 += -eta*dW3/batch_size
                bias_W3 += -eta*dbias_W3/batch_size

        #Perform Validation Set

        for t in range(0, valid_size):
            x = x_train[t]
                   
            desired_output_v = y_train[t]

            act1 = np.dot(W1, x) + bias_W1
            if sigmoid:
                # Apply the sigmoid function
                out1 = 1/(1+np.exp(-act1))
            else:
                #apply ReLU
                out1 = np.clip(act1,0.0000001,None)
            
            if n_hidden_layer2 > 0:    
                if sigmoid:
                    out2 = 1/(1+np.exp(-act2))
                else:
                    out2 = np.clip(act2,0.0000001,None)
                # Neural activation: hidden layer 1 -> hidden layer 2
                act3 = np.dot(W3,out2)+bias_W3

                # Apply the sigmoid function
                out3 = act3

            else:        
                act2 = np.dot(W2, out1) + bias_W2
                out2 = act2       

            if n_hidden_layer2 > 0:
                valid[i] = valid[i] - np.sum(desired_output_v * np.log(softmax(out3)))/x_valid.shape[0]
            else:
                valid[i] = valid[i] - np.sum(desired_output_v * np.log(softmax(out2)))/x_valid.shape[0]


        print( "Epoch ", i+1, ": error = ", errors[i])
        print( "Epoch ", i+1, ": valid = ", valid[i])


    #Calculate Accuracy
        n = x_test.shape[0]

        p_ra = 0
        correct_value = np.zeros((n,))
        predicted_value = np.zeros((n,))

        for p in range(0, n):
            x = x_test[p]
            y = y_test[p]
            
            correct_value[p] = np.argmax(y)
            
            act1 = np.dot(W1, x) + bias_W1
            out1 = 1 / (1 + np.exp(-act1))
            
            if n_hidden_layer2 > 0:
                act2 = np.dot(W2, out1) + bias_W2
                if sigmoid:    

                    out2 = 1 / (1 + np.exp(-act2))
                else:
                    out2 = np.clip(act2,0.0000001,None)
            
                act3 = np.dot(W3, out2) + bias_W3    
                out3 = act3
                
                predicted_value[p] = np.argmax(out3)
            
            else:        
                act2 = np.dot(W2, out1) + bias_W2
                out2 = act2
                predicted_value[p] = np.argmax(out2)

            if predicted_value[p] == correct_value[p]: 
                p_ra = (p_ra + 1)

        accuracy = 100*p_ra/n
        accuracy_all[i] = accuracy
        print("Accuracy = ", accuracy, '%')

    return [errors, valid, accuracy_all]

a = train_network(50, 50, 150, 100, True, 0.1, 1.0, False, False)
acc = a[2]
print(acc)
np.savetxt('150100Sigmoid.out', acc, delimiter=',')
print("lower Q", np.percentile(acc, 25, interpolation='midpoint'))
print("Upper Q", np.percentile(acc, 75, interpolation='midpoint'))
print("IQR", stats.iqr(acc, interpolation = 'midpoint'))
print("IQM")
print("Mean", np.mean(acc))

print("This was 1.0 and no noise with Sigmoid, eta 1.0, 250 layer 1, 250 layer2")
plot_err(a[0],a[1])

'''
b = train_network(50, 50, 50, 0, True, 0.05, 0.9, True, True)

plot_err(b)
'''
