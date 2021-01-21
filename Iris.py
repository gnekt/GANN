from enum import Enum
from GeneralANN.NeuralNetwork import NeuralNetwork, NeuralNetworkScope
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

class PlantCategory(Enum):
    Iris_setosa = 0
    Iris_versicolor = 1
    Iris_virginica = 2

    def __index__(self):
        return self.value

if __name__ == "__main__":
    #
    # Retrieve the data
    iris = pd.read_csv('./data/iris.csv', delimiter=',',
                       names=["sepal length", "sepal width", "petal length", "petal width", "class"], header=None
                       )
    # Just transform the string to a different class type, the one-hot encoding is inside the train method
    iris.loc[iris['class'] == "Iris-setosa", 'class'] = PlantCategory.Iris_setosa
    iris.loc[iris['class'] == "Iris-versicolor", 'class'] = PlantCategory.Iris_versicolor
    iris.loc[iris['class'] == "Iris-virginica", 'class'] = PlantCategory.Iris_virginica

    #See the data
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #ax.plot_trisurf(iris.iloc[:,3].to_numpy(), iris.iloc[:,1].to_numpy(), iris.iloc[:,2].to_numpy(), color="None")


    #How does represent a 5d plot? unfortunately we can't see them...so i thinked about a trick
    #instead of enrich the number of axis i modify the other parameters of the scatter in order to represent the other inputs

    # X = Petal Lenght
    # Y = Petal width
    # Z = Sepal Lenght
    # Size of marker = Sepal width
    # Type of class = color

    ax.scatter(iris.iloc[:, 2].to_numpy(), iris.iloc[:, 3].to_numpy(),iris.iloc[:, 0].to_numpy(),
               c=iris.iloc[:, 4].to_numpy(), s=iris.iloc[:, 1].to_numpy()*10, marker="^",label='train')








    # Define the ANN with 4 input, 3 hidden layers where each one is composed by 3 neurons, 3 outputs
    # The scope of this ANN is classification so..

    nn: NeuralNetwork = NeuralNetwork(NeuralNetworkScope.Classification,4, 3, 5, 3)

    #Show the architecture
    #nn.show_me()

    #Train it

    nn.train_network_as_Classificator(iris[:150].to_numpy(), 0.05, 5000, 3)

    #Test it

    iris_test = iris.sample(frac=1).iloc[:, 0:4].to_numpy()
    result = []
    for row in iris_test:
        print(f"My beautiful ANN sees: {row}, and choose for: {nn.predict(row)}")
        result.append(nn.predict(row))
    iris_test= np.c_[iris_test,np.asarray(result)]





    ax.scatter(iris_test[:, 2], iris_test[:, 3], iris_test[:, 0],
               c=iris_test[:, 4], s=iris_test[:, 1]*10, marker="o",label='prevision')

    ax.set_xlabel('Petal Lenght')
    ax.set_ylabel('Petal width')
    ax.set_zlabel('Sepal Lenght')
    plt.legend(loc='upper left')
    plt.show()