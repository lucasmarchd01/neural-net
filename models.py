import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        # retrun dot products of weight vector and given input
        return nn.DotProduct(self.w, x)


    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        # return 1 if the dot product is non-negative, otherwise return -1
        if nn.as_scalar(self.run(x)) >= 0:
            return 1
        else:
            return -1


    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"

        no_mistakes = False
        while not no_mistakes:
            no_mistakes = True

            # iterate through batches
            for direction, y_true in dataset.iterate_once(1):

                # get neural network prediction
                y_pred = self.get_prediction(direction)
                # convert node to floating point number
                multiplier = nn.as_scalar(y_true)

                if y_pred != multiplier:
                    # update misclassified examples if not the same
                    self.w.update(multiplier,direction)
                    no_mistakes = False


class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.batchSize = 50
        self.num_neurons_hidden_layer = 55

        # initialize weight and bias vectors
        self.weight1 = nn.Parameter(1, self.num_neurons_hidden_layer) 
        self.bias1 = nn.Parameter(1, self.num_neurons_hidden_layer)

        # Output layer
        self.weightOut = nn.Parameter(self.num_neurons_hidden_layer, 1)
        self.biasOut = nn.Parameter(1, 1)


    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        # layer 1 - compute models predictions for y
        lin_trans_1 = nn.Linear(x, self.weight1)
        predicted_y = nn.AddBias(lin_trans_1, self.bias1)
        layer_1 = nn.ReLU(predicted_y)

        # Output layer: no relu needed
        lin_trans_2 = nn.Linear(layer_1, self.weightOut)
        
        # compute and return predicted output of layer
        return nn.AddBias(lin_trans_2, self.biasOut)


    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SquareLoss(self.run(x), y)


    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"

        adjusted_rate = -0.2
        while True:

            # iterate through each batch
            for row_vect, label in dataset.iterate_once(self.batchSize):
                loss = self.get_loss(row_vect, label)
                parameters = [self.w_1, self.output_w, self.b_1, self.output_b]
                grad_1, grad_2, grad_3, grad_4 = nn.gradients(parameters, loss) 
                learning_rate = min(-0.01, adjusted_rate)

                # update the weights and bias
                self.weight1.update(learning_rate, grad_1)
                self.weightOut.update(learning_rate, grad_2)
                self.bias1.update(learning_rate, grad_3)
                self.biasOut.update(learning_rate, grad_4)

            adjusted_rate += .02
            loss = self.get_loss(nn.Constant(dataset.x), nn.Constant(dataset.y))
            if nn.as_scalar(loss) < 0.01:
                return

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.batchSize = 25             # 10
        self.hiddenLayerSize = 350     # 350
        self.classLabels = 10

        # first hidden layer 
        self.weight1 = nn.Parameter(784, self.hiddenLayerSize)
        self.bias1 = nn.Parameter(1, self.hiddenLayerSize)

        # second hidden layer
        self.weight2 = nn.Parameter(self.hiddenLayerSize, self.hiddenLayerSize)
        self.bias2 = nn.Parameter(1, self.hiddenLayerSize)

        # third hidden layer
        self.weight3 = nn.Parameter(self.hiddenLayerSize, self.hiddenLayerSize)
        self.bias3 = nn.Parameter(1, self.hiddenLayerSize)

        # output layer
        self.weightOut = nn.Parameter(self.hiddenLayerSize, self.classLabels)
        self.biasOut = nn.Parameter(1, self.classLabels)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        layer1 = nn.ReLU(nn.AddBias(nn.Linear(x, self.weight1), self.bias1))
        layer2 = nn.ReLU(nn.AddBias(nn.Linear(layer1, self.weight2), self.bias2))
        layer3 = nn.ReLU(nn.AddBias(nn.Linear(layer2, self.weight3), self.bias3))
        layerOut = nn.AddBias(nn.Linear(layer3, self.weightOut), self.biasOut)
        return layerOut


    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        # return the softmax loss
        return nn.SoftmaxLoss(self.run(x), y)


    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        
    
        adjustRate = -0.1
        doneTraining = 0

        while doneTraining < 0.98:

            for row_vect, y in dataset.iterate_once(self.batchSize):
                loss = self.get_loss(row_vect, y)
                parameters = ([self.weight1, self.weight2, self.weight3, self.weightOut, self.bias1, self.bias2, self.bias3, self.biasOut])
                g1, g2, g3, g4, g5, g6, g7, g8 = nn.gradients(parameters, loss) 
                learningRate = min(-0.005, adjustRate)

                # updates
                self.weight1.update(learningRate, g1)
                self.weight2.update(learningRate, g2)
                self.weight3.update(learningRate, g3)
                self.weightOut.update(learningRate, g4)
                self.bias1.update(learningRate, g5)
                self.bias2.update(learningRate, g6)
                self.bias3.update(learningRate, g7)
                self.biasOut.update(learningRate, g8)

            adjustRate += 0.05
            # check for 98 % accuracy after each epoch, not after each batch
            doneTraining = dataset.get_validation_accuracy()


            '''
            
            batchSize = 100
            loss = float('inf')
            validationAccuracy = 0

            while validationAccuracy < 0.98:

                for x, y in dataset.iterate_once(batchSize):
                    loss = self.get_loss(x, y)
                    gradients = nn.gradients(loss, self.parameters)
                    loss = nn.as_scalar(loss)
                    for i in range(len(self.parameters)):
                        self.parameters[i].update(gradients[i], -self.lr)
                validationAccuracy = dataset.get_validation_accuracy()
            '''