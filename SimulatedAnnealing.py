from sklearn.neural_network import MLPClassifier
import random
import numpy as np
import pandas
import copy
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

df = pandas.read_csv('red-wine.csv')

df = df.values
y = df[:,11]
X = df[:,0:11]
X = scale(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

def noop():
    pass

def get_neighbor(candidate):
    neighbor = copy.deepcopy(candidate)
    for j in range(len(neighbor)):
        for k in range(len(neighbor[j])):
            for m in range(len(neighbor[j][k])):
                randomNum = random.randint(0, 2)
                randFloat = random.random()
                if (randomNum == 0):
                    neighbor[j][k][m] += (randFloat * 0.01)
                if (randomNum == 1):
                    neighbor[j][k][m] -= (randFloat * 0.01)
    return neighbor

def score(candidate, x, y):
    classifier.coefs_ = candidate
    return classifier.score(x, y)

# initialize classifier
classifier = MLPClassifier(solver='none', alpha=1e-5, hidden_layer_sizes=(100, ), random_state=157, max_iter=0)
classifier._validate_hyperparameters = noop
classifier = classifier.fit(X_train, y_train)


simulatedAnnealingBestAccuracy = classifier.score(X_train, y_train)
simulatedAnnealingBestSuccessor = classifier.coefs_
print(simulatedAnnealingBestSuccessor)
iterationBestSuccessor = []
numIters = []

# Initialize temperature
temperature = 0.1
temperature_min = 0.0001
alpha = 0.9999
badIterations = 0

for i in range(1000):
    # find the successor
    successor = get_neighbor(simulatedAnnealingBestSuccessor)
    accuracy = score(successor, X_train, y_train)
    print(accuracy)
    if accuracy > simulatedAnnealingBestAccuracy:
        simulatedAnnealingBestAccuracy = accuracy
        simulatedAnnealingBestSuccessor = successor
    else:
        if temperature > temperature_min:
            acceptanceProb = math.exp((accuracy - simulatedAnnealingBestAccuracy) / temperature)
            #print(acceptanceProb)
            randomNum = random.random()
            #print(randomNum)
            if acceptanceProb > randomNum:
                simulatedAnnealingBestAccuracy = accuracy
                simulatedAnnealingBestSuccessor = successor
    iterationBestSuccessor.append(simulatedAnnealingBestSuccessor)
    temperature *= alpha

#####
randomSeed = []
errors = list()
test_errors = list()
for i in range(len(iterationBestSuccessor)):
    classifier.coefs_ = iterationBestSuccessor[i]
    print("Accuracy: %f" % classifier.score(X_train, y_train))
    errors.append(classifier.score(X_train, y_train))
    test_errors.append(classifier.score(X_test, y_test))
    randomSeed.append(i)

i_iter_optim = np.argmax(errors)
alpha_optim = randomSeed[i_iter_optim]
print("Optimal iteration num: %s" % alpha_optim)
classifier.coefs_ = iterationBestSuccessor[alpha_optim]
print("Optimal random seed accuracy: %f" % classifier.score(X_test, y_test))
# Plot results functions

plt.subplot(2, 1, 1)
plt.plot(randomSeed, errors, label='Training Data Set')
plt.plot(randomSeed, test_errors, label='Test Data Set')
plt.vlines(alpha_optim, plt.ylim()[0], np.max(test_errors), color='k',
        linewidth=3, label='Optimum on test')
plt.legend(loc='upper left')
plt.ylim([0.4, 1.1])
plt.xlabel('Iteration')
plt.ylabel('Prediction Accuracy')
plt.title('Simulated Annealing')
plt.show()