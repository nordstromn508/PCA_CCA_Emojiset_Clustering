"""
main.py
    Main thread of execution

    @author Nicholas Nordstrom
"""
import data_loader
import image_processing
import numpy as np
import matplotlib.pyplot as plt


def main():
    X, y, centers = data_loader.generate_dataset(n_clusters=4, random_state=42)
    plt.scatter(X[:, 0], X[:, 1])
    plt.show()


if __name__ == "__main__":
    main()