Project 2 - Randomized Optimization

Required data:
    The only required dataset is included in the repository as red-wine.csv.

Running my code:
    The code for each of the 3 discrete optimization problems is contained in a file named after the associated problem: countones.py, fourpeaks.py, and salesman.py.
    You'll need python 3, mlrose(along with mlrose's dependencies such as numpy), and matplotlib to run these files. In order to run them, simply run the command 'python <file>.py' or 
    'python3 <file>.py' depending on your environment. This will create all the graphs shown in the analysis. Note that the number of iterations is varied in the 
    analysis but held constant in the python scripts - to change the number of iterations, simply change the max_iters parameter in the calls to the mlrose algorithms.

    The code for the neural net weight search problem is contained in three different python files: RandomizedHillClimbing.py, SimulatedAnnealing.py, and GeneticAlgorithm.py. Simply use the command 
    'python <file>.py' (or 'python3 <file>.py'). These will also generate the plots shown in the analysis.

Attributions:
    The implementation of the three algorithms used for neural net weight finding was implemented by a previous student, Maya Pogrebinsky, and modified by me for purposes of my 
    assignment. The code for the three discrete optimization problems was written solely by me.

The source code can be found at https://github.com/jmeyers35/CS4641-Randomized-Optimization.