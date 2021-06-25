import numpy as np
import csv
import sys
import json
import sys

from validate import validate

train_X_file_path = "train_X_nb.csv"
train_Y_file_path = "train_Y_nb.csv"
alpha = 1
validation_split = 0.2

def trainValSplit(X,Y):
    train_X = np.copy(X)
    train_Y = np.copy(Y)
    valIndex = -int(validation_split*(train_X.shape[0]))
    val_X = train_X[valIndex:]
    val_Y = train_Y[valIndex:]
    train_X = train_X[:valIndex]
    train_Y = train_Y[:valIndex]
    return (train_X, train_Y, val_X, val_Y)

def preprocessing(Ss):
    outS = []
    for s in Ss:
        # out = ""
        # for c in s:
        #     if (c == " " and len(out) != 0 and out[-1] != " ") or c.isalpha():
        #         out += c.lower()
        out = s.lower()
        outS.append(out.strip())
    return outS

def class_wise_words_frequency_dict(X, Y):
    Ys = np.unique(Y)
    out = dict()
    for y in Ys:
        wordCount = dict()
        Xs = np.array(X)[np.array(Y)==y]
        for s in Xs:
            for w in s.split(" "):
                if w not in wordCount.keys():
                    wordCount[w] = 1 
                else:
                    wordCount[w] += 1
        out[int(y)] = wordCount
    return out

def compute_prior_probabilities(Y):
    Ys = np.unique(Y)
    out = {}
    for y in Ys:
        out[int(y)] = np.sum(np.array(Y) == y)/len(Y)
    return out

def get_class_wise_denominators_likelihood(X, Y, alpha=1):
    Ys = np.unique(Y)
    out = {}
    classWiseWordCount = class_wise_words_frequency_dict(X, Y)
    vocab = []
    for clas in classWiseWordCount.keys():
        vocab.extend(list(classWiseWordCount[clas].keys()))
    vocab = list(set(vocab))
    for y in Ys:
        out[int(y)] = int(np.sum(list(classWiseWordCount[y].values())) + alpha*len(vocab))
    return out 

def compute_likelihood(test_X, c, class_wise_frequency_dict, class_wise_denominators, alpha=1):
    sum = 0
    for w in test_X.split(" "):
        count = class_wise_frequency_dict[c].get(w)
        if count == None:
            count = 0
        sum += np.log((count+alpha)/class_wise_denominators[c])
    return sum

def predictClass(test_X, class_wise_frequency_dict, class_wise_denominators, prior_probabilities, alpha=1):
    maxClass = -1
    maxPosterior = -np.inf
    for clas in class_wise_frequency_dict.keys():
        curLikelihood = compute_likelihood(test_X, clas, class_wise_frequency_dict, class_wise_denominators, alpha)
        curPosterior = np.log(prior_probabilities[clas]) + curLikelihood
        if curPosterior > maxPosterior:
            maxPosterior = curPosterior 
            maxClass = clas 
    return maxClass

"""
Predicts the target values for data in the file at 'test_X_file_path', using the model learned during training.
Writes the predicted values to the file named "predicted_test_Y_nb.csv". It should be created in the same directory where this code file is present.
This code is provided to help you get started and is NOT a complete implementation. Modify it based on the requirements of the project.
"""

def import_data_and_model(test_X_file_path, model_file_path):
    test_X = np.genfromtxt(test_X_file_path, delimiter='\n', dtype=str)
    model = json.load(open(model_file_path))
    return test_X, model


def predict_target_values(test_X, model):
    # Write your code to Predict Target Variables
    # HINT: You can use other functions which you've already implemented in coding assignments.
    Y = []
    for x in test_X:
        Y.append(predictClass(x, model["class_wise_words_frequency"], model["class_wise_denominators_likelihood"], model["prior_probabilities"], model["alpha"]))
    return np.array(Y)


def write_to_csv_file(pred_Y, predicted_Y_file_name):
    pred_Y = pred_Y.reshape(len(pred_Y), 1)
    with open(predicted_Y_file_name, 'w', newline='') as csv_file:
        wr = csv.writer(csv_file)
        wr.writerows(pred_Y)
        csv_file.close()


def predict(test_X_file_path):
    # Load Model Parameters
    """
    Parameters of Naive Bayes include Laplace smoothing parameter, Prior probabilites of each class and values related to likelihood computation.
    """
    if "-trainModel" in sys.argv:
        train_X = np.genfromtxt(train_X_file_path, delimiter="\n", skip_header=0,dtype=str)
        train_Y = np.genfromtxt(train_Y_file_path, delimiter="\n", skip_header=0,dtype=int)
        train_X = preprocessing(train_X)
        train_X, train_Y, val_X, val_Y = trainValSplit(train_X, train_Y)
        
        bestModel = None
        maxAcc = 0
        for alpha in [0.0075, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.4, 0.8, 1]:
            model = {}
            model["class_wise_words_frequency"] = class_wise_words_frequency_dict(train_X, train_Y)
            model["prior_probabilities"] = compute_prior_probabilities(train_Y)
            model["class_wise_denominators_likelihood"] = get_class_wise_denominators_likelihood(train_X, train_Y, alpha)
            model["alpha"] = alpha
            acc = np.sum(np.array(val_Y) == predict_target_values(val_X, model))/len(val_Y)
            print(f"alpha={alpha}, acc={acc}")
            if acc >= maxAcc:
                bestModel = model
                maxAcc = acc
        with open("./MODEL_FILE.json","w") as fp:
            json.dump(bestModel, fp)

    test_X, model = import_data_and_model(test_X_file_path, "./MODEL_FILE.json")
    pred_Y = predict_target_values(test_X, model)
    write_to_csv_file(pred_Y, "predicted_test_Y_nb.csv")    


if __name__ == "__main__":
    test_X_file_path = sys.argv[1]
    predict(test_X_file_path)
    # Uncomment to test on the training data
    validate(test_X_file_path, actual_test_Y_file_path="train_Y_nb.csv") 