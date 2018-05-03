import sys
import math
import numpy as np

train_input = sys.argv[1] 
index_to_word = sys.argv[2] 
index_to_tag = sys.argv[3] 
hmmprior = sys.argv[4] 
hmmemit = sys.argv[5] 
hmmtrans = sys.argv[6] 

def get_train_data():
    with open(train_input, 'rb') as fd:
        input = fd.read().strip('\n').split('\n')
        for i in range(len(input)):
            input[i] = input[i].replace('_', ' ').split(' ')
        X = [row[::2] for row in input]
        Y = [row[1::2] for row in input]
    return (X,Y)

def get_tag_dict():
    with open(index_to_tag, 'rb') as fd:
        input = fd.read().strip('\n').split('\n')
        return {k:v for v,k in enumerate(input)}

def get_word_dict():
    with open(index_to_word, 'rb') as fd:
        input = fd.read().strip('\n').split('\n')
        return {k:v for v,k in enumerate(input)}

def get_priors(Y,tag_dict):
    priors = [0] * len(tag_dict)
    for row in Y:
        priors[tag_dict[row[0]]] += 1
    sum = np.float64(0.0)
    for prior in priors:
        sum += prior+1
    priors = map(lambda p: (p+1)/sum, priors)

    with open(hmmprior, 'w') as fd:
        for entry in priors:
            fd.write(str(entry)+'\n')

    return priors

def get_transitions(priors,Y,tag_dict):
    transitions = [[0 for i in range(len(tag_dict))] for j in range(len(tag_dict))]
    for i in range(len(Y)):
        for j in range(len(Y[i])-1):
            current = Y[i][j+1]
            prev = Y[i][j]
            transitions[tag_dict[prev]][tag_dict[current]] += 1
    for i in range(len(transitions)):
        row = transitions[i]
        sum = np.float64(0.0)
        for entry in row:
            sum += entry+1
        transitions[i] = map(lambda x: (x+1)/sum,row)

    with open(hmmtrans, 'w') as fd:
        for entry in transitions:
            entry = map(lambda x: str(x), entry)
            row = " ".join(entry)
            fd.write(row+'\n')

    return transitions

def get_emissions(X,Y,word_dict,tag_dict):
    emissions = [[0 for i in range(len(word_dict))] for j in range(len(tag_dict))]
    for i in range(len(X)):
        for j in range(len(X[i])):
            row = tag_dict[Y[i][j]]
            col = word_dict[X[i][j]]
            emissions[row][col] += 1
    for i in range(len(emissions)):
        row = emissions[i]
        sum = np.float64(0.0)
        for entry in row:
            sum += entry+1
        emissions[i] = map(lambda x: (x+1)/sum,row)

    with open(hmmemit, 'w') as fd:
        for entry in emissions:
            entry = map(lambda x: str(x), entry)
            row = " ".join(entry)
            fd.write(row+'\n')

    return emissions

def main():
    (X,Y) = get_train_data()
    tag_dict = get_tag_dict()
    word_dict = get_word_dict()
    priors = get_priors(Y,tag_dict)
    transitions = get_transitions(priors,Y,tag_dict)
    emissions = get_emissions(X,Y,word_dict,tag_dict)

if __name__ == "__main__":
    main()
