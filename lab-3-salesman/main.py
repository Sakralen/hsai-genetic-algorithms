import sys
import time

from fileutils import parse_csv
from salesmangenetic import SalesmanGeneticAlgoritm

# import plotutils
# import routeutils
# import os
# import matplotlib.pyplot as plt

if __name__ == "__main__":
    population_size = int(sys.argv[1])
    genereation_max = int(sys.argv[2])
    crossover_prob = float(sys.argv[3])
    mutation_prob = float(sys.argv[4])

    locations = parse_csv('csv/eil51.csv')
    # optimal_route_classic = [int(item) - 1 for row in parse_csv(
    #     'csv/eil51-best-route.csv') for item in row]
    # optimal_route_neighbour = routeutils.to_neighbour_route(optimal_route_classic)
    #
    # optimal_length = routeutils.calc_route_length(optimal_route_neighbour, locations)
    #
    # path = os.path.join(os.path.dirname(__file__), 'output/')
    # if not os.path.exists(path):
    #     os.makedirs(path)
    #
    # plt.xlabel("X")
    # plt.ylabel("Y")
    # plt.title(f'Best route; length: {optimal_length:0.0f}')
    #
    # plotutils.plot_locations(locations)
    # plotutils.plot_restored_route(optimal_route_classic, locations)
    #
    # plt.savefig(
    #     path + 'eil51_optimal.png')

    start_time = time.time()

    SalesmanGeneticAlgoritm(locations, population_size,
                            genereation_max, crossover_prob, mutation_prob).execute()

    finish_time = time.time() - start_time
    print(f'elapsed time: {finish_time:0.2f}')
