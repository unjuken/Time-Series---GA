import pygad
import numpy 

"""
Given the following function:
    y = f(w1:w6) = w1x1 + w2x2 + w3x3 + w4x4 + w5x5 + 6wx6
    where (x1,x2,x3,x4,x5,x6)=(4,-2,3.5,5,-11,-4.7) and y=44
What are the best values for the 6 weights (w1 to w6)? We are going to use the genetic algorithm to optimize this function.
"""
class GeneticAlgorithm:

    def __init__(self, models, observations, numberOfGenerations):
        
        num_rows, num_cols = models.shape

        def fitness_func(solution, solution_idx):
            # Calculating the fitness value of each solution in the current population.
            # The fitness function calulates the sum of products between each input and its corresponding weight.
            w = numpy.matrix(solution)
            SEI = 0
            i = 0
            for observation in observations:
                predictedObservation = (models[i]*w.T).item()
                maxV = predictedObservation if predictedObservation > observation else observation 
                minV = predictedObservation if predictedObservation < observation else observation
                SEI += minV/maxV
                i+=1
            #endfor
            #fitness = 1.0 / (observations.size - SEI)
            fitness = SEI
            return fitness
        #endFitFunc

        fitness_function = fitness_func

        num_generations = numberOfGenerations # Number of generations.
        num_parents_mating = 7 # Number of solutions to be selected as parents in the mating pool.

        # To prepare the initial population, there are 2 ways:
        # 1) Prepare it yourself and pass it to the initial_population parameter. This way is useful when the user wants to start the genetic algorithm with a custom initial population.
        # 2) Assign valid integer values to the sol_per_pop and num_genes parameters. If the initial_population parameter exists, then the sol_per_pop and num_genes parameters are useless.
        sol_per_pop = 50 # Number of solutions in the population.
        num_genes = num_cols #genes equals the number of models.

        init_range_low = 0
        init_range_high = 1

        parent_selection_type = "sss" # Type of parent selection.
        keep_parents = 7 # Number of parents to keep in the next population. -1 means keep all parents and 0 means keep nothing.

        crossover_type = "single_point" # Type of the crossover operator.

        # Parameters of the mutation operation.
        mutation_type = "random" # Type of the mutation operator.
        mutation_percent_genes = 80 # Percentage of genes to mutate. This parameter has no action if the parameter mutation_num_genes exists or when mutation_type is None.

        last_fitness = 0
        def callback_generation(ga_instance):
            nonlocal last_fitness
            print("Generation = {generation} Fitness = {fitness} Change = {change}".format(generation=ga_instance.generations_completed, fitness=ga_instance.best_solution()[1], change=ga_instance.best_solution()[1] - last_fitness))
            last_fitness = ga_instance.best_solution()[1]

        # Creating an instance of the GA class inside the ga module. Some parameters are initialized within the constructor.
        ga_instance = pygad.GA(num_generations=num_generations,
                            num_parents_mating=num_parents_mating, 
                            fitness_func=fitness_function,
                            sol_per_pop=sol_per_pop, 
                            num_genes=num_genes,
                            init_range_low=init_range_low,
                            init_range_high=init_range_high,
                            random_mutation_min_val=init_range_low,
                            random_mutation_max_val=init_range_high,
                            parent_selection_type=parent_selection_type,
                            keep_parents=keep_parents,
                            crossover_type=crossover_type,
                            mutation_type=mutation_type,
                            mutation_percent_genes=mutation_percent_genes,
                            callback_generation=callback_generation)

        # Running the GA to optimize the parameters of the function.
        ga_instance.run()

        # After the generations complete, some plots are showed that summarize the how the outputs/fitenss values evolve over generations.
        #ga_instance.plot_result()

        # Returning the details of the best solution.
        self.solution, self.solution_fitness, self.solution_idx = ga_instance.best_solution()
        print("Parameters of the best solution : {solution}".format(solution=self.solution))
        print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=self.solution_fitness))
        print("Index of the best solution : {solution_idx}".format(solution_idx=self.solution_idx))

        #print("Predicted output based on the best solution : {prediction}".format(prediction=self.prediction))

        if ga_instance.best_solution_generation != -1:
            print("Best fitness value reached after {best_solution_generation} generations.".format(best_solution_generation=ga_instance.best_solution_generation))
    #endInit
#endClass



