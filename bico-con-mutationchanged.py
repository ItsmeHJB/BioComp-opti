# GA Python Script


import random
from operator import attrgetter
import math
import matplotlib.pyplot as plt


class Individual:
    gene = []
    fitness = 0


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

        temp_pop.append((min(candidates, key=attrgetter("fitness"))))

    return temp_pop

#haixia
# # Roulette-wheel selection
# def roulette_selection(roulette_pop):
#     temp_pop = []
#     for rou_i in range(len(roulette_pop)):
#         total_fit = calc_total_fitness(roulette_pop)
#         randfit = random.uniform(total_fit, 0)
#         # randfit = random.random() % calc_total_fitness(roulette_pop)
#         running_total = 0
#         index = 0
#         while running_total >= randfit:
#             running_total += population[index].fitness
#             index += 1

#         temp_pop.append(population[index-1])

#     return temp_pop
#haixia
def roulette_selection(pop,num): #select num individuals
  selected = []
  sum_of_fitness = sum([ind.fitness for ind in pop])
  previous_probability = 0
  list_shares = []
  for i in pop:
    current_share = previous_probability + i.fitness/sum_of_fitness
    list_shares.append(current_share)
    previous_probability = current_share
  #now lets spin the wheel:
  array_shares = np.array(list_shares)
  print(array_shares)
  for j in range(0,num):    
    random_number = random.uniform(0,max(list_shares))
    print('random_number:')
    print(random_number)
    array_idx = np.where(array_shares >= random_number)# can do it in another way:
    #array_idx = np.where(array_shares <= random_number)
    #selected.append(pop[array_idx[0][len(array_idx)-1]])
    if len(array_idx)>0:
      if len(array_idx[0])>0:
        selected.append(pop[array_idx[0][0]])
      else:
        print('len(array_idx[0]) = ' + str(len(array_idx[0])))
    else:
        print('len(array_idx) = ' + str(len(array_idx)))
  return selected


def single_point_crossover(ind1, ind2):
    crossover_point = random.randint(1, gene_size - 1)
    ind1.gene[crossover_point:], ind2.gene[crossover_point:] = ind2.gene[crossover_point:], ind1.gene[crossover_point:]

    return ind1, ind2



def bitwise_mutation(indiv):
    newind = Individual()
    newind.gene = []
    for j in range(0, gene_size):
        chromosome = indiv.gene[j]
        if random.random() < mutation_rate:
            alter = random.uniform(0, mutation_step)
            # if random.random() % 2:#haixia: what are you trying to achieve? print random.random() % 2 out, see what you get
            #     chromosome = (chromosome + alter) % upper_limit
            if random.randint(0,1)==1:#haixia: similating flipping a coin: if you get front end, you do addition 
                chromosome = chromosome + alter
                if chromosome > upper_limit:
                    chromosone = upper_limit
            else:
                chromosome = (chromosome - alter)
                if chromosome < lower_limit:
                    # chromosome = chromosome + upper_limit#haixia
                    chromosone = lower_limit

        newind.gene.append(chromosome)
    newind.fitness = eval_gene_new(newind.gene)
    return newind

#haixia
# def bitwise_mutation(indiv):
#     newind = Individual()
#     newind.gene = []
#     for j in range(0, gene_size):
#         chromosome = indiv.gene[j]
#         if random.random() < mutation_rate:
#             alter = random.uniform(0, mutation_step)
#             if random.random() % 2:
#                 chromosome = (chromosome + alter) % upper_limit
#             else:
#                 chromosome = (chromosome - alter)
#                 if chromosome < lower_limit:
#                     chromosome = chromosome + upper_limit

#         newind.gene.append(chromosome)
#     newind.fitness = eval_gene_w3(newind.gene)
#     return newind


def generate_fitness_for_pop(fitness_pop):
    for fit_i in range(len(fitness_pop)):
        fitness_pop[fit_i].fitness = 0
        for fit_j in range(len(fitness_pop[0].gene)):
            fitness_pop[fit_i].fitness += fitness_pop[fit_i].gene[fit_j]


def calc_total_fitness(total_fit_pop):
    total_fit = 0
    for fit_i in range(len(total_fit_pop)):
        total_fit += total_fit_pop[fit_i].fitness

    return total_fit


def find_best_fitness(best_fit_pop):
    current_best = float("inf")
    for best_i in range(len(best_fit_pop)):
        if best_fit_pop[best_i].fitness < current_best:
            current_best = best_fit_pop[best_i].fitness

    return current_best


def calc_mean_fitness(mean_fit_pop):
    total_mean_fit = 0
    for mean_i in range(len(mean_fit_pop)):
        total_mean_fit += mean_fit_pop[mean_i].fitness

    avg_fitness = total_mean_fit / len(mean_fit_pop)
    return avg_fitness


# def calc_min_fitness(min_fit_pop):
#     current_min = float("inf")
#     for min_i in range(len(min_fit_pop)):
#         if min_fit_pop[min_i].fitness < current_min:
#             current_min = min_fit_pop[min_i].fitness
#
#     return current_min


def eval_gene_w3(gene):
    # f(x) = 10n + sum(x[i]^2 – 10 * cos(2 * pi * x[i]))
    # Where n = gene_size, x = gene
    fitness = 10 * gene_size
    for i in range(gene_size):
        chromosome = gene[i]
        lhs = chromosome**2
        rhs = 10 * math.cos(2 * math.pi * chromosome)
        fitness = fitness + (lhs - rhs)

    return fitness


def eval_gene_new(gene):
    # f(x) = -20 * exp(-0.2 * sqrt(sum(x**2)/genelen)) - exp(sum(cos(2*pi*x))/genelen)
    LHB = 0
    for i in range(gene_size):
        LHB += gene[i]**2
    LHB = LHB / gene_size
    LHB = math.sqrt(LHB)
    LHB = LHB * -0.2

    RHB = 0
    for i in range(gene_size):
        RHB += math.cos(2 * math.pi * gene[i])
    RHB = RHB / gene_size

    fitness = -20 * math.exp(LHB) - math.exp(RHB)
    return fitness


def create_init_pop(p):
    temp_pop = []

    for x in range(0, p):
        temp_gene = []
        for y in range(0, gene_size):
            temp_gene.append(random.uniform(lower_limit, upper_limit))
        newind = Individual()
        newind.gene = temp_gene.copy()
        fitness = eval_gene_w3(temp_gene)
        newind.fitness = fitness

        temp_pop.append(newind)

    return temp_pop


def create_next_generation(pop_to_change):
    next_gen = []
    for i in range(0, pop_size, 2):
        child1, child2 = single_point_crossover(pop_to_change[i], pop_to_change[i+1])
        child1 = bitwise_mutation(child1)
        child2 = bitwise_mutation(child2)
        next_gen.append(child1)
        next_gen.append(child2)

    return next_gen


# Initialise constants
gene_size = 20
pop_size = 50
mutation_rate = 1/pop_size
tourney_size = 5
number_of_generations = 50
upper_limit = 32#haixia
lower_limit = -32#haixia
mutation_step = random.uniform(0, upper_limit)
total_runs = 50


if __name__ == '__main__':
    best_scores = []
    mean_scores = []
    generations = []
    # Make list for use in graphs
    for i in range(number_of_generations):
        generations.append(i)

    for run_num in range(total_runs):
        population = create_init_pop(pop_size)

        # total_fitness = []
        best_gen_fitness = []
        mean_gen_fitness = []
        # Run for set number of generations
        for generation_number in range(number_of_generations):
            parents = roulette_selection(population)
            # parents = tourney_selection(population, tourney_size)
            population = create_next_generation(parents)
            population = roulette_selection(population)
            # population = tourney_selection(population, tourney_size)

            best_gen_fitness.append(find_best_fitness(population))
            mean_gen_fitness.append(calc_mean_fitness(population))

        best_scores.append(best_gen_fitness)
        mean_scores.append(mean_gen_fitness)

    best_avg_fitness = []
    mean_avg_fitness = []
    # Avg values across runs for each generation
    # Squash [10,200] -> [200]
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

    plt.plot(generations, best_avg_fitness, color='g')
    plt.plot(generations, mean_avg_fitness, color='orange')
    plt.title(
        "Best in gen (green) with avg fitness (orange) over " + str(number_of_generations) + " gens across " + str(
            total_runs) + " runs")
    plt.show()