import matplotlib.pyplot as plt
import mlrose
import numpy as np
import time


def initialize_problem(n):
    problem = mlrose.DiscreteOpt(n, fitness_fn=mlrose.FourPeaks(), maximize=True, max_val=2)
    return problem

def simulated_annealing():
    print("STARTING FOUR PEAKS PROBLEM WITH SIMULATED ANNEALING")
    fitnesses = []
    runtime = []
    for n in range(10, 110, 10):
        start = time.time()
        problem = initialize_problem(n)
        schedule = mlrose.ExpDecay()
        np.random.seed(65)
        init_state = np.random.randint(2, size=n)

        # Solve
        best_state, best_fitness = mlrose.simulated_annealing(problem, schedule=schedule, max_attempts=10, max_iters=100, init_state=init_state)
        finish_time = time.time() - start
        runtime.append(finish_time)
        fitnesses.append(best_fitness)
        print('time for simulated annealing for n={}: {}'.format(n, time.time() - start))
        #print('best state for n={}: {}'.format(n, best_state))
        print('best fitness for n={}: {}'.format(n, best_fitness))
    return fitnesses, runtime

def rhc():
    print("STARTING FOUR PEAKS PROBLEM WITH RANDOMIZED HILL CLIMBING")
    fitnesses = []
    runtime = []
    for n in range(10, 110, 10):
        start = time.time()
        problem = initialize_problem(n)
        np.random.seed(65)
        init_state = np.random.randint(2, size=n)

        # Solve
        best_state, best_fitness = mlrose.random_hill_climb(problem, max_attempts=10, max_iters=100, init_state=init_state)
        finish_time = time.time() - start
        runtime.append(finish_time)
        fitnesses.append(best_fitness)
        print('time for rhc for n={}: {}'.format(n, time.time() - start))
        #print('best state for n={}: {}'.format(n, best_state))
        print('best fitness for n={}: {}'.format(n, best_fitness))
    return fitnesses, runtime


def genetic_algorithm():
    print("STARTING FOUR PEAKS PROBLEM WITH GENETIC ALGORITHM")
    fitnesses = []
    runtime = []
    for n in range(10, 110, 10):
        start = time.time()
        problem = initialize_problem(n)
        schedule = mlrose.ExpDecay()
        np.random.seed(65)
        init_state = np.random.randint(2, size=n)
        # Solve
        best_state, best_fitness = mlrose.genetic_alg(problem, max_attempts=10, max_iters=100)
        finish_time = time.time() - start
        runtime.append(finish_time)
        fitnesses.append(best_fitness)
        print('time for genetic algorithm for n={}: {}'.format(n, time.time() - start))
        #print('best state for n={}: {}'.format(n, best_state))
        print('best fitness for n={}: {}'.format(n, best_fitness))
    return fitnesses, runtime

def mimic():
    print("STARTING FOUR PEAKS PROBLEM WITH MIMIC")
    fitnesses = []
    runtime = []
    for n in range(10, 110, 10):
        start = time.time()
        problem = initialize_problem(n)
        np.random.seed(65)
        init_state = np.random.randint(2, size=n)
        # Solve
        best_state, best_fitness = mlrose.mimic(problem, max_attempts=10, max_iters=100)
        finish_time = time.time() - start
        runtime.append(finish_time)
        fitnesses.append(best_fitness)
        print('time for MIMIC for n={}: {}'.format(n, time.time() - start))
        #print('best state for n={}: {}'.format(n, best_state))
        print('best fitness for n={}: {}'.format(n, best_fitness))
    return fitnesses, runtime


def main():
    sa_fitness, sa_runtime = simulated_annealing()
    rhc_fitness, rhc_runtime = rhc()
    ga_fitness, ga_runtime = genetic_algorithm()
    mimic_fitness, mimic_runtime = mimic()
    n_bits = [n for n in range(10, 110, 10)]

    plt.figure()
    plt.title('Randomized Optimization Algorithm Performance: Four Peaks (n_iterations=100)')
    plt.xlabel('Length of Input (Number of Bits)')
    plt.ylabel('Best Fitness')
    plt.plot(n_bits, sa_fitness, label='Simulated Annealing')
    plt.plot(n_bits, rhc_fitness, label='Randomized Hill-Climbing')
    plt.plot(n_bits, ga_fitness, label='Genetic Algorithm')
    plt.plot(n_bits, mimic_fitness, label='MIMIC')
    plt.legend(loc='upper left')
    plt.show()

    plt.figure()
    plt.title('Randomized Optimization Algorithm Runtime: Four Peaks (n_iterations=100)')
    plt.xlabel('Length of Input (Number of Bits)')
    plt.ylabel('Runtime (Seconds)')
    plt.plot(n_bits, sa_runtime, label='Simulated Annealing')
    plt.plot(n_bits, rhc_runtime, label='Randomized Hill-Climbing')
    plt.plot(n_bits, ga_runtime, label='Genetic Algorithm')
    plt.plot(n_bits, mimic_runtime, label='MIMIC')
    plt.legend(loc='upper left')
    plt.show()


if __name__ == "__main__":
    main()