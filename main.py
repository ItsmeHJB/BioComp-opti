# GA Python Script - Harrison Bennion 17012546
import random
from operator import attrgetter
import math
import matplotlib.pyplot as plt
import numpy as np


# Class to hold genes and fitness
class Individual:
    gene = []
    fitness = 0
    sigma = []  # Used in gaussian mutation


# K way tournament selection
#   Pick K people at random
#   Return individual with highest fitness
def tourney_selection(tourney_pop, k):
    temp_pop = []
    for tourn_i in range(len(tourney_pop)):
        candidates = []
        for index in range(k):
            random_int = random.randint(0, len(tourney_pop) - 1)
            candidates.append(tourney_pop[random_int])
        # Change to max(...) is doing maximise function
        temp_pop.append((min(candidates, key=attrgetter("fitness"))))

    return temp_pop


# Roulette wheel selection - returns a pop of size num
def roulette_selection(roulette_pop, num):
    selected = []
    sum_of_fitness = calc_total_fitness(roulette_pop)
    previous_probability = 0
    list_shares = []
    # Gather list of fitness/total fit for pop
    for rou_i in roulette_pop:
        current_share = previous_probability + rou_i.fitness / sum_of_fitness
        list_shares.append(current_share)
        previous_probability = current_share
    array_shares = np.array(list_shares)
    # Spin the wheel to pick a value
    for rou_i in range(num):
        random_num = random.uniform(0, max(list_shares))
        # Swap round if maximising
        array_idx = np.where(array_shares >= random_num)
        # array_idx = np.where(array_shares <= random_num)
        if len(array_idx) > 0:
            if len(array_idx[0]) > 0:
                selected.append(roulette_pop[array_idx[0][0]])

    return selected


# Single-point crossover - Pick a point and swap tails of genes
def single_point_crossover(ind1, ind2):
    crossover_point = random.randint(1, gene_size - 1)
    ind1.gene[crossover_point:], ind2.gene[crossover_point:] = ind2.gene[crossover_point:], ind1.gene[crossover_point:]

    return ind1, ind2


# Uniform crossover - flip a coin for each chromosome
def uniform_crossover(ind1, ind2):
    for cross_i in range(gene_size):
        if random.randint(0, 1) == 1:
            # Swap chromosomes inline
            ind1.gene[cross_i], ind2.gene[cross_i] = ind2.gene[cross_i], ind1.gene[cross_i]

    return ind1, ind2


# Mutate each chromosome based on mutation rate
def bitwise_mutation(indiv):
    newind = Individual()
    newind.gene = []
    for muti_i in range(gene_size):
        chromosome = indiv.gene[muti_i]
        if random.random() < mutation_rate:
            # Generate an alteration value
            alter = random.uniform(0, mutation_step)
            # 'Flip a coin' to add or subtract
            if random.randint(0, 1) == 1:
                # Ensure it's within the limits
                chromosome = (chromosome + alter) % upper_limit
            else:
                chromosome = (chromosome - alter)
                # Ensure it's within limits
                if chromosome < lower_limit:
                    chromosome = lower_limit

        newind.gene.append(chromosome)
    newind.fitness = eval_gene_new(newind.gene)
    return newind


# Randomly alter chromosomes to be vals defined between the lower and upper limit
def uniform_mutation(indiv):
    newind = Individual()
    newind.gene = []
    for mut_i in range(gene_size):
        chromosome = indiv.gene[mut_i]
        if random.random() < mutation_rate:
            # Set chromosone to random value between limits
            chromosome = random.uniform(lower_limit, upper_limit)

        newind.gene.append(chromosome)
    newind.fitness = eval_gene_new(newind.gene)
    return newind


# Non-uniform mutation
def non_uniform_mutation(indiv):
    newind = Individual()
    newind.gene = indiv.gene.copy()
    rand_i = random.randint(0, gene_size-1)

    chromosome = newind.gene[rand_i]
    # Toss a coin to start at lower, or upper limit
    if random.random():
        chromosome = (upper_limit-chromosome) * non_uni_func(generation_number)
    else:
        chromosome = (lower_limit+chromosome) * non_uni_func(generation_number)

    newind.gene[rand_i] = chromosome
    newind.fitness = eval_gene_new(newind.gene)
    return newind


# Mutation step function used in non-uniform mutation
def non_uni_func(current_gen_num):
    return random.random() * (1 - current_gen_num/number_of_generations)


# Not working unfortunately - Converges in a positive direction?
# Gaussian mutation attempt
def gaussian_mutation(indiv):
    newind = Individual()
    newind.gene = []
    for mut_i in range(gene_size):
        # Calc std deviation from sigma val
        stddev = max(0, indiv.sigma[mut_i] * math.exp(gauss_func(0, 1)))

        # Do gauss func on chromosome using stddev
        chromosome = indiv.gene[mut_i]
        chromosome = gauss_func(chromosome, stddev)

        # Check within bounds
        if chromosome >= upper_limit:
            chromosome = upper_limit
        elif chromosome <= lower_limit:
            chromosome = lower_limit

        # Append chromosome and new sigma value
        newind.gene.append(chromosome)
        newind.sigma.append(stddev)
    newind.fitness = eval_gene_new(newind.gene)
    return newind


# Gaussian function (DO NOT CALL IN MAIN CODE)
def gauss_func(mean, stddev):
    x1 = random.uniform(0, 1)
    x2 = random.uniform(0, 1)

    y1 = math.sqrt(-2.0 * math.log(x1, 10)) * math.cos(2.0 * math.pi * x2)
    return y1 * stddev + mean


# Calc total fitness of pop
def calc_total_fitness(total_fit_pop):
    total_fit = 0
    for fit_i in range(len(total_fit_pop)):
        total_fit += total_fit_pop[fit_i].fitness

    return total_fit


# Finds best fitness from pop
def find_best_fitness(best_fit_pop):
    current_best = float("inf")
    for best_i in range(len(best_fit_pop)):
        if best_fit_pop[best_i].fitness < current_best:
            current_best = best_fit_pop[best_i].fitness

    return current_best


# Calc mean fitness of pop
def calc_mean_fitness(mean_fit_pop):
    total_mean_fit = 0
    for mean_i in range(len(mean_fit_pop)):
        total_mean_fit += mean_fit_pop[mean_i].fitness

    avg_fitness = total_mean_fit / len(mean_fit_pop)
    return avg_fitness


# Worksheet 3 optimisation problem
def eval_gene_w3(gene):
    # f(x) = 10n + sum(x[i]^2 – 10 * cos(2 * pi * x[i]))
    # Where n = gene_size, x = gene
    fitness = 10 * gene_size
    for eval_i in range(gene_size):
        chromosome = gene[eval_i]
        lhs = chromosome**2
        rhs = 10 * math.cos(2 * math.pi * chromosome)
        fitness = fitness + (lhs - rhs)

    return fitness


# New problem from optimisation worksheet
def eval_gene_new(gene):
    # f(x) = -20 * exp(-0.2 * sqrt(sum(x**2)/genelen)) - exp(sum(cos(2*pi*x))/genelen)
    # -0.2 * sqrt(sum(x**2)/genelen)
    lhb = 0
    for eval_i in range(gene_size):
        lhb += gene[eval_i]**2
    lhb = lhb / gene_size
    lhb = math.sqrt(lhb)
    lhb = lhb * -0.2

    # sum(cos(2*pi*x))/genelen
    rhb = 0
    for eval_i in range(gene_size):
        rhb += math.cos(2 * math.pi * gene[eval_i])
    rhb = rhb / gene_size

    # f(x) = -20 * exp(lhb) - exp(rhb)
    fitness = -20 * math.exp(lhb) - math.exp(rhb)
    return fitness


# Create initial population for each run
def create_init_pop(p):
    temp_pop = []

    for x in range(0, p):
        newind = Individual()
        temp_gene = []
        sigma_start = (upper_limit - lower_limit) / 6
        # Set chromosomes in genes to random reals between limits
        for y in range(0, gene_size):
            temp_gene.append(random.uniform(lower_limit, upper_limit))
            newind.sigma.append(sigma_start)  # Only used in gaussian mutation
        newind.gene = temp_gene.copy()
        # Update eval function if working on new problem
        fitness = eval_gene_new(temp_gene)
        newind.fitness = fitness

        temp_pop.append(newind)

    return temp_pop


# Crossover + mutation population
def create_next_generation(pop_to_change):
    next_gen = []
    # Iterate in steps of 2 due to crossover func
    for create_i in range(0, pop_size, 2):
        # Choose single_point or uniform crossover
        child1, child2 = single_point_crossover(pop_to_change[create_i], pop_to_change[create_i+1])
        # Choose bitwise, uniform, or non_uniform mutation
        # Gaussian does not work currently
        child1 = bitwise_mutation(child1)
        child2 = bitwise_mutation(child2)
        next_gen.append(child1)
        next_gen.append(child2)

    return next_gen


# Initialise constants
gene_size = 20
pop_size = 500
mutation_rate = 0.02
tourney_size = 2
number_of_generations = 150
upper_limit = 32.0
lower_limit = -32.0
total_runs = 10
mutation_step = random.uniform(upper_limit/3, upper_limit)
print(mutation_step)


# Main program
if __name__ == '__main__':
    best_scores = []
    mean_scores = []
    generations = []
    # Make list for use in graphs
    for i in range(number_of_generations):
        generations.append(i)

    # Complete set number of runs
    for run_num in range(total_runs):
        population = create_init_pop(pop_size)

        best_gen_fitness = []
        mean_gen_fitness = []
        # Run for set number of generations
        for generation_number in range(number_of_generations):
            # Select type of selection algorithm
            parents = tourney_selection(population, tourney_size)
            # Call method to crossover + mutate population
            population = create_next_generation(parents)
            # Change selection method here too
            population = tourney_selection(population, tourney_size)

            # Store best + mean fits, call total fit if needed
            best_gen_fitness.append(find_best_fitness(population))
            mean_gen_fitness.append(calc_mean_fitness(population))

        # Add to larger array for averaging across runs
        best_scores.append(best_gen_fitness)
        mean_scores.append(mean_gen_fitness)

    best_avg_fitness = []
    mean_avg_fitness = []
    # Avg values across runs for each generation
    # e.g. Squash [10,200] -> [200]
    for i in range(number_of_generations):
        best_sum = 0
        mean_sum = 0
        for j in range(total_runs):
            best_sum += best_scores[j][i]
            mean_sum += mean_scores[j][i]
        best_sum = best_sum / total_runs
        mean_sum = mean_sum / total_runs
        best_avg_fitness.append(best_sum)
        mean_avg_fitness.append(mean_sum)

    # Plot best and mean fitness' against generations
    plt.plot(generations, best_avg_fitness, color="forestgreen", label="Best fit")
    plt.plot(generations, mean_avg_fitness, color="darkviolet", label="Mean fit")
    plt.title("GA: " + str(number_of_generations) + " generations, averaged over " + str(total_runs) + " runs")
    plt.ylabel("Fitness")
    plt.xlabel("Generations")
    plt.legend(loc="best")
    plt.show()
