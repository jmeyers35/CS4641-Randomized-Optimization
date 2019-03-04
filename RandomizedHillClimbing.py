from sklearn.neural_network import MLPClassifier
import random
import numpy as np
import pandas
import copy
import matplotlib.pyplot as plt
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

def get_neighbors(candidate):
    neighbors = []
    choices = [0, -0.1, 0.1]
    for i in range(10):
        random.seed(i * 233)
        candidateCopy = copy.deepcopy(candidate)
        for j in range(len(candidateCopy)):
            for k in range(len(candidateCopy[j])):
                for m in range(len(candidateCopy[j][k])):
                    randomNum = random.randint(0, 2)
                    if (randomNum == 0):
                        candidateCopy[j][k][m] += choices[randomNum]
                    if (randomNum == 1):
                        candidateCopy[j][k][m] += choices[randomNum]
                    else:
                        candidateCopy[j][k][m] += choices[randomNum]
        neighbors.append(candidateCopy)
    return neighbors

def score(candidate, x, y):
    classifier.coefs_ = candidate
    return classifier.score(x, y)


hillClimbBestAccuracy = 0
hillClimbBestSuccessor = []

iterationBestSuccessor = []
numIters = []

# 100 random restarts (see random_state=x)
for x in range(100):
    # initialize classifier
    classifier = MLPClassifier(solver='none', alpha=1e-5, hidden_layer_sizes=(100, ), random_state=x, max_iter=0)
    classifier._validate_hyperparameters = noop
    classifier = classifier.fit(X_train, y_train)
    
    # initialize iteration args
    hillClimbIterBestSuccessor = classifier.coefs_
    hillClimbIterBestAccuracy = classifier.score(X_train, y_train)
    badIterations = 0
    
    # hill climb 100 times
    for i in range(1000):
        successors = get_neighbors(hillClimbIterBestSuccessor)
        bestSuccessorAccuracy = hillClimbIterBestAccuracy
        bestSuccessor = hillClimbIterBestSuccessor
        # find the best successor
        for j in range(len(successors)):
            accuracy = score(successors[j], X_train, y_train)
            if accuracy > bestSuccessorAccuracy:
                bestSuccessorAccuracy = accuracy
                bestSuccessor = successors[j]
        if bestSuccessorAccuracy < hillClimbIterBestAccuracy + 0.000000000001 and badIterations > 5:
            print("iteration {}".format(i))
            numIters.append(i)
            break
        if bestSuccessorAccuracy < hillClimbIterBestAccuracy + 0.000000000001:
            badIterations += 1
        hillClimbIterBestAccuracy = bestSuccessorAccuracy
        hillClimbIterBestSuccessor = bestSuccessor
        #print("Best Accuracy:", bestSuccessorAccuracy)
    if hillClimbIterBestAccuracy > hillClimbBestAccuracy:
        hillClimbBestAccuracy = hillClimbIterBestAccuracy
        hillClimbBestSuccessor = hillClimbIterBestSuccessor
    print hillClimbIterBestAccuracy
    iterationBestSuccessor.append(hillClimbIterBestSuccessor)

#####
classifier = MLPClassifier(solver='none', alpha=1e-5, hidden_layer_sizes=(100, ), random_state=x, max_iter=0)
classifier._validate_hyperparameters = noop
classifier = classifier.fit(X, y)
randomSeed = []
errors = list()
test_errors = list()
for i in range(100):
    classifier.coefs_ = iterationBestSuccessor[i]
    print("Accuracy: %f" % classifier.score(X_train, y_train))
    errors.append(classifier.score(X_train, y_train))
    test_errors.append(classifier.score(X_test, y_test))
    randomSeed.append(i)

i_iter_optim = np.argmax(errors)
alpha_optim = randomSeed[i_iter_optim]
print("Optimal random seed parameter : %s" % alpha_optim)
classifier.coefs_ = iterationBestSuccessor[alpha_optim]
print("Optimal random seed accuracy: %f" % classifier.score(X_test, y_test))
# Plot results functions

plt.subplot(2, 1, 1)
plt.plot(randomSeed, errors, label='Training Data Set')
plt.plot(randomSeed, test_errors, label='Test Data Set')
plt.vlines(alpha_optim, plt.ylim()[0], np.max(test_errors), color='k',
        linewidth=3, label='Optimum on test')
plt.legend(loc='lower left')
plt.ylim([0.4, 1.1])
plt.xlabel('Random Seed')
plt.ylabel('Prediction Accuracy')
plt.title('Randomized Hill Climbing')
plt.show()


plt.subplot(2, 1, 1)
plt.plot(randomSeed, numIters, label='Data Set')
plt.vlines(alpha_optim, plt.ylim()[0], np.max(errors), color='k',
        linewidth=3, label='Max Iters')
plt.legend(loc='lower left')
plt.ylim([0, 50])
plt.xlabel('Random Seed')
plt.ylabel('Number of Iterations')
plt.title('Randomized Hill Climbing')
plt.show()