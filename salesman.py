import matplotlib.pyplot as plt
import mlrose
import numpy as np
import time
import random



def initialize_problem(n):
    random.seed(420)
    points = [(random.randint(0,50),random.randint(0,50)) for _ in range(n)]
    #print(points)
    fitness = mlrose.TravellingSales(coords=points)
    problem = mlrose.TSPOpt(n, fitness_fn=fitness, coords=points)
    return problem

def simulated_annealing():
    print("STARTING SALESMAN PROBLEM WITH SIMULATED ANNEALING")
    fitnesses = []
    runtime = []
    for n in range(5, 26, 1):
        start = time.time()
        problem = initialize_problem(n)
        schedule = mlrose.ExpDecay()
        np.random.seed(65)
        init_state = np.array([node for node in range(n)])

        # Solve
        best_state, best_fitness = mlrose.simulated_annealing(problem, schedule=schedule, max_attempts=10, max_iters=100, init_state=None)
        finish_time = time.time() - start
        runtime.append(finish_time)
        fitnesses.append(-1 * best_fitness)
        print('time for simulated annealing for n={}: {}'.format(n, time.time() - start))
        #print('best state for n={}: {}'.format(n, best_state))
        print('best fitness for n={}: {}'.format(n, best_fitness))
    return fitnesses, runtime

def rhc():
    print("STARTING SALESMAN PROBLEM WITH RANDOMIZED HILL CLIMBING")
    fitnesses = []
    runtime = []
    for n in range(5, 26, 1):
        start = time.time()
        problem = initialize_problem(n)
        np.random.seed(65)
        init_state = np.array([node for node in range(n)])
        # Solve
        best_state, best_fitness = mlrose.random_hill_climb(problem, max_attempts=10, max_iters=100, init_state=None)
        finish_time = time.time() - start
        runtime.append(finish_time)
        fitnesses.append(-1 * best_fitness)
        print('time for rhc for n={}: {}'.format(n, time.time() - start))
        #print('best state for n={}: {}'.format(n, best_state))
        print('best fitness for n={}: {}'.format(n, best_fitness))
    return fitnesses, runtime


def genetic_algorithm():
    print("STARTING SALESMAN PROBLEM WITH GENETIC ALGORITHM")
    fitnesses = []
    runtime = []
    for n in range(5, 26, 1):
        start = time.time()
        problem = initialize_problem(n)
        schedule = mlrose.ExpDecay()
        np.random.seed(65)
        # Solve
        best_state, best_fitness = mlrose.genetic_alg(problem, max_attempts=10, max_iters=100)
        finish_time = time.time() - start
        runtime.append(finish_time)
        fitnesses.append(-1 * best_fitness)
        print('time for genetic algorithm for n={}: {}'.format(n, time.time() - start))
        #print('best state for n={}: {}'.format(n, best_state))
        print('best fitness for n={}: {}'.format(n, best_fitness))
    return fitnesses, runtime

def mimic():
    print("STARTING SALESMAN PROBLEM WITH MIMIC")
    fitnesses = []
    runtime = []
    for n in range(5, 26, 1):
        start = time.time()
        problem = initialize_problem(n)
        np.random.seed(65)

        # Solve
        best_state, best_fitness = mlrose.mimic(problem,max_attempts=10, max_iters=100)
        finish_time = time.time() - start
        runtime.append(finish_time)
        fitnesses.append(-1 * best_fitness)
        print('time for MIMIC for n={}: {}'.format(n, time.time() - start))
        #print('best state for n={}: {}'.format(n, best_state))
        print('best fitness for n={}: {}'.format(n, best_fitness))
    return fitnesses, runtime

def main():
    sa_fitness, sa_runtime = simulated_annealing()
    rhc_fitness, rhc_runtime = rhc()
    ga_fitness, ga_runtime = genetic_algorithm()
    mimic_fitness, mimic_runtime = mimic()
    n_nodes = [n for n in range(5, 26, 1)]

    plt.figure()
    plt.title('Randomized Optimization Algorithm Performance: Travelling Salesman (n_iterations=100)')
    plt.xlabel('Length of Input (Number of Vertices)')
    plt.ylabel('Best Fitness')
    plt.plot(n_nodes, sa_fitness, label='Simulated Annealing')
    plt.plot(n_nodes, rhc_fitness, label='Randomized Hill-Climbing')
    plt.plot(n_nodes, ga_fitness, label='Genetic Algorithm')
    plt.plot(n_nodes, mimic_fitness, label='MIMIC')
    #plt.gca().invert_yaxis()
    plt.legend(loc='upper left')
    plt.show()

    plt.figure()
    plt.title('Randomized Optimization Algorithm Runtime: Travelling Salesman (n_iterations=100)')
    plt.xlabel('Length of Input (Number of Vertices)')
    plt.ylabel('Runtime (Seconds)')
    plt.plot(n_nodes, sa_runtime, label='Simulated Annealing')
    plt.plot(n_nodes, rhc_runtime, label='Randomized Hill-Climbing')
    plt.plot(n_nodes, ga_runtime, label='Genetic Algorithm')
    plt.plot(n_nodes, mimic_runtime, label='MIMIC')
    plt.legend(loc='upper left')
    plt.show()


if __name__ == "__main__":
    main()