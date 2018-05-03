import sys
import math
import numpy as np

test_input = sys.argv[1]
index_to_word = sys.argv[2] 
index_to_tag = sys.argv[3] 
hmmprior = sys.argv[4]
hmmemit = sys.argv[5] 
hmmtrans = sys.argv[6] 
predicted_file = sys.argv[7]

def get_test_data():
    with open(test_input, 'rb') as fd:
        input = fd.read().strip('\n').split('\n')
        for i in range(len(input)):
            input[i] = input[i].replace('_', ' ').split(' ')
        X = [row[::2] for row in input]
        Y = [row[1::2] for row in input]
    return (X,Y)

def get_tag_dict():
    with open(index_to_tag, 'rb') as fd:
        input = fd.read().strip('\n').split('\n')
        return {v:k for v,k in enumerate(input)}

def get_word_dict():
    with open(index_to_word, 'rb') as fd:
        input = fd.read().strip('\n').split('\n')
        return {k:v for v,k in enumerate(input)}

def get_priors():
    with open(hmmprior, 'rb') as fd:
        input = fd.read().strip('\n').split('\n')
    return input

def get_emissions():
    with open(hmmemit, 'rb') as fd:
        input = fd.read().strip('\n').split('\n')
    for i in range(len(input)):
        input[i] = input[i].split(' ')
    return input

def get_transitions():
    with open(hmmtrans, 'rb') as fd:
        input = fd.read().strip('\n').split('\n')
    for i in range(len(input)):
        input[i] = input[i].split(' ')
    return input

def get_alphas(priors,emissions,transitions,sentence,word_len,tag_len,word_dict):
    alphas = np.zeros((word_len,tag_len))
    for ((i,j),val) in np.ndenumerate(alphas):
        if (i==0):
            alphas[i][j] = priors[j]*emissions[j][word_dict[sentence[i]]]
        else:
            sum = np.dot(alphas[i-1,:],transitions[:,j])
            alphas[i][j] = emissions[j][word_dict[sentence[i]]]*sum
    return alphas

def get_betas(emissions,transitions,sentence,word_len,tag_len,word_dict):
    betas = np.zeros((word_len,tag_len))
    for ((i,j),val) in np.ndenumerate(betas):
        row = word_len-1-i
        if (row==(word_len-1)):
            betas[row][j] = 1
        else:
            sum = 0
            word = word_dict[sentence[row+1]]
            for k in range(tag_len):
                sum += emissions[k][word]*betas[row+1][k]*transitions[j][k]
            betas[row][j] = sum
    return betas

def predict(X,predictions):
    combos = list(X)
    for i in range(len(combos)):
        for j in range(len(combos[i])):
            combos[i][j] = str(X[i][j])+"_"+str(predictions[i][j])
    with open(predicted_file, 'w') as fd:
        for row in combos:
            row = " ".join(row)
            fd.write(row+'\n')

def main():
    (X,Y) = get_test_data()
    tag_dict = get_tag_dict()
    word_dict = get_word_dict()
    tag_len = len(tag_dict)
    priors = get_priors()
    priors = np.asarray(priors,dtype=float)
    emissions = get_emissions()
    emissions = np.asarray(emissions,dtype=float)
    transitions = get_transitions()
    transitions = np.asarray(transitions,dtype=float)

    predictions = list(Y)
    for i in range(len(X)):
        sentence = X[i]
        word_len = len(sentence)
        alphas = get_alphas(priors,emissions,transitions,sentence,word_len,tag_len,word_dict)
        betas = get_betas(emissions,transitions,sentence,word_len,tag_len,word_dict)
        for j in range(len(sentence)):
            prediction = np.argmax(np.multiply(alphas[j],betas[j]))
            predictions[i][j] = tag_dict[prediction]
    predict(X,predictions)

if __name__ == "__main__":
    main()
