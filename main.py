"""
main.py
    Main thread of execution

    @author Nicholas Nordstrom
"""
from random import randint
import data_loader
import image_processing
import models
import evaluation
import numpy as np
import matplotlib.pyplot as plt

RANDOM_STATE = 42
NUM_DATASETS = 10
MAX_N_CLUSTERS = 3
MIN_N_CLUSTERS = 2
MAX_STD = 1
MIN_STD = 0
MAX = 20
MIN = -20


def main():

    # Generate Datasets
    params = []  # n_clusters, std, min, max
    for n in range(NUM_DATASETS):
        params.append([randint(MIN_N_CLUSTERS, MAX_N_CLUSTERS), randint(MIN_STD*10+1, MAX_STD*10)/10.0, MIN, MAX])
        print("PARAM {}: ".format(n), params[n])

    datasets = [data_loader.generate_dataset(n_clusters=p[0], std=p[1], min=p[2], max=p[3], random_state=RANDOM_STATE) for p in params]

    # Visualize Datasets
    for i in range(len(datasets)):
        X = datasets[i][0]
        plt.scatter(X[:, 0], X[:, 1])
        plt.show()


if __name__ == "__main__":
    main()
