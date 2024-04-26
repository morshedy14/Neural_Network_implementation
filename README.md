The Neural Network (NN) implementation in this project represents a versatile, custom-built feedforward neural network framework developed using Python. It is designed to provide a foundational understanding of neural network operations, including layer addition, activation functions, forward propagation, and back propagation.

### Key Features:

- **Layer Flexibility**: Users can dynamically add layers to the neural network, specifying the number of neurons and the type of activation function (ReLU or Sigmoid) for each layer. This feature allows for the construction of both simple and complex network architectures tailored to specific tasks.
- **Custom Activation Functions**: The implementation includes commonly used activation functions—ReLU for non-linear transformations retaining positive values and Sigmoid for output layers in binary classification tasks. These functions are crucial for introducing non-linearity to the model, enabling it to learn more complex patterns in the data.
- **Forward and Back Propagation**: During the forward pass, the network calculates the output for a given input, while back propagation adjusts the network’s parameters (weights and biases) based on the error gradient. This training process iteratively improves the model's accuracy on the training data.
- **Parameter Optimization**: Using gradient descent, the network updates its parameters to minimize the loss function. This approach is fundamental for learning from the training data and refining the model's predictions.
- **Modular Design**: The neural network is structured in a way that each component (layers, activations, weight updates) is self-contained, enhancing the readability and maintainability of the code. This modular design also facilitates experimenting with different network configurations and parameters.

### Implementation Details:

- **Initialization**: The neural network class initializes with an empty list of layers and dictionaries to store parameters, gradients, and intermediate values during computation.
- **Adding Layers**: Users can add layers to the network, automatically linking them based on their order of addition, which simplifies the setup for multi-layer networks.
- **Training and Prediction**: The network is trained using a specified number of epochs and learning rate, which can be adjusted to optimize performance. After training, the network can make predictions on new data, demonstrating its utility in practical applications.

This NN implementation not only serves as an educational tool for those new to neural networks but also as a robust framework capable of tackling real-world classification tasks by adjusting its complexity and tuning its parameters.



### Features

- **Custom Neural Network Class**: Includes a neural network class with methods for adding layers, forward propagation, back propagation, training, and prediction.
- **Custom Layer Implementation**: Ability to create layers with specified number of neurons and activation functions, including ReLU and Sigmoid.
- **Dataset Application**: Implementation demonstrated on the Iris dataset, complete with data preprocessing and a descriptive analysis.
- **Model Training and Evaluation**: Detailed training process with adjustable epochs and learning rate, alongside evaluation using accuracy metrics.

### Prerequisites

This project requires Python 3.x and the following Python libraries installed:
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

You can install these packages with pip:
```sh
pip install numpy pandas matplotlib seaborn scikit-learn
```
