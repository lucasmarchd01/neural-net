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
        # retrun dot product of weight vector and given input
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

        # while there are still mistakes
        while not no_mistakes:
            no_mistakes = True

            # iterate through batches
            for x,y in dataset.iterate_once(1):

                # get neural network prediction
                y_pred = self.get_prediction(x)

                # convert node to floating point number and check if it is equal to true label
                if y_pred != nn.as_scalar(y):

                    # update misclassified examples if not the same
                    self.w.update(nn.as_scalar(y),x)
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
        
        # initialize LR and batch size
        self.learningRate = .008
        self.batchSize = 100

        # layer 1 parameters
        self.weight1 = nn.Parameter(1, 128)
        self.bias1 = nn.Parameter(1, 128)

        # layer 2 parameters
        self.weight2 = nn.Parameter(128, 64)
        self.bias2 = nn.Parameter(1, 64)

        # output layer parameters
        self.weightOut = nn.Parameter(64, 1)
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

        # layer 1 output
        layer1 = nn.ReLU(nn.AddBias(nn.Linear(x, self.weight1), self.bias1))

        # layer 2 output
        layer2 = nn.ReLU(nn.AddBias(nn.Linear(layer1, self.weight2), self.bias2))
        
        # compute and return predicted output of last layer
        layerOut = nn.AddBias(nn.Linear(layer2, self.weightOut), self.biasOut)
        return layerOut
  

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
        
        parameters = [self.weight1, self.bias1, self.weight2, self.bias2, self.weightOut, self.biasOut]
        
        while True:
            # iterate though the batches
            for x, y in dataset.iterate_once(self.batchSize):
                # get softmax loss
                loss = self.get_loss(x, y)
                # get gradients of the loss with respect to the paraneters
                gradients = nn.gradients(parameters, loss)

                # update the weights and biases with the learning rate and biases
                for i in range(len(parameters)):
                    parameters[i].update(-self.learningRate, gradients[i])
            
            # check if loss is less than 0.02, if so exit loop
            if nn.as_scalar(loss) < 0.02:
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
        self.learningRate = 0.1
        self.batchSize = 100

        # first hidden layer
        self.weight1 = nn.Parameter(784, 256)
        self.bias1 = nn.Parameter(1, 256)

        # second hidden layer
        self.weight2 = nn.Parameter(256, 128)
        self.bias2 = nn.Parameter(1, 128)

        # third hidden layer
        self.weight3 = nn.Parameter(128, 64)
        self.bias3 = nn.Parameter(1, 64)

        # output layer
        self.weightOut = nn.Parameter(64, 10)
        self.biasOut = nn.Parameter(1, 10)


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

        # layer 1 output 
        layer1 = nn.ReLU(nn.AddBias(nn.Linear(x, self.weight1), self.bias1))

        # layer 2 output 
        layer2 = nn.ReLU(nn.AddBias(nn.Linear(layer1, self.weight2), self.bias2))

        # layer 3 output
        layer3 = nn.ReLU(nn.AddBias(nn.Linear(layer2, self.weight3), self.bias3))

        # compute and return output layer
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
        
        validationAccuracy = 0
        parameters = ([self.weight1, self.weight2, self.weight3, self.weightOut, self.bias1, self.bias2, self.bias3, self.biasOut])
        
        # iterate while the validation accuracy is less than 98%
        while validationAccuracy < 0.98:

            # iterate through the batches
            for x,y in dataset.iterate_once(self.batchSize):

                # get the softmax loss
                loss = self.get_loss(x, y)
                # get gradients of the loss with respect to the paraneters
                gradients = nn.gradients(parameters, loss)
                
                # update the weights and biases with the learning rate and gradients
                for i in range(len(parameters)):
                    parameters[i].update(-self.learningRate, gradients[i])

            # get the validation accuracy
            validationAccuracy = dataset.get_validation_accuracy()

        
