import pandas as pd
import  numpy as np
import csv
from GeneralANN.NeuralNetwork import NeuralNetwork, NeuralNetworkScope
from matplotlib import  pyplot as plt


if __name__ == "__main__":

    covid = pd.read_csv('./data/dpc-covid19-ita-andamento-nazionale.csv', delimiter=',')  # Pick the 11° column
    Train = covid.loc[:, ["ricoverati_con_sintomi", "terapia_intensiva", "isolamento_domiciliare", "dimessi_guariti",
                       "deceduti"]]

    Train = Train / Train.max()

    # Define a NN
    nn: NeuralNetwork = NeuralNetwork(NeuralNetworkScope.Regression,4, 1, 10, 1)

    # Train the NN
    nn.train_network_as_Regression(Train.to_numpy(), 1, 100) #without bias

    # Test the NN
    X_prevision = Train.loc[:, "ricoverati_con_sintomi":"dimessi_guariti"]
    Y_prevision = []
    for row in X_prevision.to_numpy():
        Y_prevision.append(nn.predict(row))

    plt.plot(range(0, Train.shape[0]), Train.loc[:,"deceduti"].to_numpy(), "r^", range(0, len(Y_prevision)), Y_prevision, "k")
    plt.show()