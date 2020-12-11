import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from baseline import BasicBaseline
from utils.basics import load_model, test_confusion_matrix, plot_confusion_matrix

def main():
    model = BasicBaseline()
    load_model(model.model, "defense_cm")
    # load_model(model.model, "nodefense_cm")

    model.load_data()
    cm = test_confusion_matrix(model.model, model.testloader)
    print(cm)
    cm_plt = plot_confusion_matrix(cm, model.trainset.classes, normalize=True)
    cm_plt.savefig("figures/defense_cm.png")
    # cm_plt.savefig("figures/nodefense_cm.png")
    




if __name__ == "__main__":
    main()