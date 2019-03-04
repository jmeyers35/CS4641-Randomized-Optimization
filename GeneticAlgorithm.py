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

def create_next_generation(curGeneration):
    nextGeneration = []
    for i in range(len(curGeneration) / 4):
        parent1Index = random.randint(0, len(curGeneration) - 1)
        parent2Index = random.randint(0, len(curGeneration) - 1)

        jSplit = random.randint(0, len(curGeneration[i]) - 1)
        kSplit = random.randint(0, (len(curGeneration[i][jSplit]) - 1) % 11)
        mSplit = random.randint(0, len(curGeneration[i][jSplit][kSplit]) - 1)
        
        randomNum = random.randint(0, 1)
        mutationProbability = random.random()
        randFloat = random.random()

        child1 = copy.deepcopy(curGeneration[parent1Index])
        child2 = copy.deepcopy(curGeneration[parent2Index])

        for j in range(jSplit):
            for k in range(kSplit):
                for m in range(mSplit):
                    print('j: {}, k: {}, m: {}'.format(j,k,m))
                    child1[j][k][m] = curGeneration[parent1Index][j][k][m]
                    child2[j][k][m] = curGeneration[parent2Index][j][k][m]
                    randomNum = random.randint(0, 1)
                    mutationProbability = random.random()
                    randFloat = random.random()
                    if (mutationProbability > 0.7):
                        if (randomNum == 0):
                            child1[j][k][m] += (randFloat * 0.01)
                            child2[j][k][m] -= (randFloat * 0.01)
                        if (randomNum == 1):
                            child1[j][k][m] -= (randFloat * 0.01)
                            child2[j][k][m] += (randFloat * 0.01)

        for j in range(len(curGeneration[i]) - jSplit - 1):
            for k in range(len(curGeneration[i][j]) - kSplit - 1):
                for m in range(len(curGeneration[i][j][k]) - mSplit - 1):
                    child1[j][k][m] = curGeneration[parent1Index][j][k][m]
                    child2[j][k][m] = curGeneration[parent2Index][j][k][m]
                    randomNum = random.randint(0, 1)
                    mutationProbability = random.random()
                    randFloat = random.random()
                    if (mutationProbability > 0.3):
                        if (randomNum == 0):
                            child1[j][k][m] += (randFloat * 0.01)
                            child2[j][k][m] -= (randFloat * 0.01)
                        if (randomNum == 1):
                            child1[j][k][m] -= (randFloat * 0.01)
                            child2[j][k][m] += (randFloat * 0.01)
        nextGeneration.append(child1)
        nextGeneration.append(child2)
    while len(nextGeneration) < len(curGeneration):
        randParentIndex = random.randint(0, len(curGeneration) - 1)
        child = copy.deepcopy(curGeneration[randParentIndex])
        for j in range(len(child)):
            for k in range(len(child[j])):
                for m in range(len(child[j][k])):
                    randomNum = random.randint(0, 2)
                    randFloat = random.random()
                    if (randomNum == 0):
                        child[j][k][m] += (randFloat * 0.01)
                    if (randomNum == 1):
                        child[j][k][m] -= (randFloat * 0.01)
        nextGeneration.append(child)
    return nextGeneration

def score(candidate, x, y):
    classifier.coefs_ = candidate
    return classifier.score(x, y)

def create_first_generation(k):
    firstGeneration = []
    for i in range(k):
        classifier = MLPClassifier(solver='none', alpha=1e-5, hidden_layer_sizes=(100, ), random_state=k, max_iter=0)
        classifier._validate_hyperparameters = noop
        classifier = classifier.fit(X_train, y_train)
        firstGeneration.append(classifier.coefs_)
    return firstGeneration

# initialize classifier
classifier = MLPClassifier(solver='none', alpha=1e-5, hidden_layer_sizes=(100, ), random_state=157, max_iter=0)
classifier._validate_hyperparameters = noop
classifier = classifier.fit(X_train, y_train)

gABestAccuracy = 0
gABestSuccessor = []

iterationBestSuccessor = []
numIters = []
genSize = 100
curGeneration = create_first_generation(genSize)
sortedList = sorted(curGeneration, key=lambda x: score(x, X_train, y_train))

# 100 iterations
for x in range(1000):
    # find the successor
    nextGeneration = create_next_generation(curGeneration)
    
    twoGenerations = curGeneration + nextGeneration
    sortedList = sorted(twoGenerations, key=lambda x: score(x, X_train, y_train), reverse=True)
    curGeneration = sortedList[:100]

    iterationBestSuccessor.append(curGeneration[0])
    print(score(curGeneration[0], X_train, y_train))



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
print("Optimal accuracy: %f" % classifier.score(X, y))
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
plt.title('Genetic Algorithm')
plt.show()


# plt.subplot(2, 1, 1)
# plt.plot(randomSeed, numIters, label='Data Set')
# plt.vlines(alpha_optim, plt.ylim()[0], np.max(errors), color='k',
#         linewidth=3, label='Max Iters')
# plt.legend(loc='lower left')
# plt.ylim([0, 50])
# plt.xlabel('Random Seed')
# plt.ylabel('Number of Iterations')
# plt.title('Simulated Annealing')
# plt.show()