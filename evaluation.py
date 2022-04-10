"""
evaluation.py
    A home for different model evaluation methods

    @author: Nicholas Nordstrom
"""
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix as conf_mat, ConfusionMatrixDisplay


def confusion_matrix(predictions, ground_truth, plot=True):
    """
    Creates a confusion matrix and prints it to the command line
    :param predictions: predictions from the model
    :param ground_truth: ground truth labels matching predictions in order
    :param plot: boolean value to enable plotting and visualization of confusion matrix
    :return: 0 if ran without error
    """
    cm = conf_mat(ground_truth, predictions)
    print("Confusion Matrix:\n {}".format(cm))

    if plot:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.show()
    return 0
