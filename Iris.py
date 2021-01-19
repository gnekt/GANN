from enum import Enum
from GeneralANN.NeuralNetwork import NeuralNetwork, NeuralNetworkScope
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

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
    ax.plot_trisurf(iris.iloc[:,0].to_numpy(), iris.iloc[:,1].to_numpy(), iris.iloc[:,2].to_numpy(), color="None")
    ax.scatter(iris.iloc[:,2].to_numpy(), iris.iloc[:,3].to_numpy(), iris.iloc[:,0].to_numpy(), c=iris.iloc[:,4].to_numpy() , s=100)
    ax.set_xlabel('Petal Lenght')
    ax.set_ylabel('Petal width')
    ax.set_zlabel('Sepal Lenght')


    plt.show()




    # Define the ANN with 4 input, 3 hidden layers where each one is composed by 3 neurons, 3 outputs
    # The scope of this ANN is classification so..
    nn: NeuralNetwork = NeuralNetwork(NeuralNetworkScope.Classification,4, 3, 5, 3)

    #Show the architecture
    #nn.show_me()

    #Train it
    nn.train_network_as_Classificator(iris.to_numpy(), 0.05, 1000, 3)

    #Test it
    iris_test = iris.iloc[:, 0:4].to_numpy()
    for row in iris_test:
        print(f"My beautiful ANN sees: {row}, and choose for: {nn.predict(row)}")

