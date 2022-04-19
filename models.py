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
        self.batch_size = 25
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
        lin_trans_1 = nn.Linear(x, self.weight_1)
        predicted_y = nn.AddBias(lin_trans_1, self.bias_1)
        layer_1 = nn.ReLU(predicted_y)

        # Output layer: no relu needed
        lin_trans_2 = nn.Linear(layer_1, self.output_w)
        
        # compute and return predicted output of layer
        return nn.AddBias(lin_trans_2, self.output_b)


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

            for row_vect, label in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(row_vect, label)
                params = [self.w_1, self.output_w, self.b_1, self.output_b]
                gradients = nn.gradients(params, loss) # (loss, params)
                learning_rate = min(-0.01, adjusted_rate)

                # updates
                self.w_1.update(learning_rate, gradients[0])
                self.output_w.update(learning_rate, gradients[1])
                self.b_1.update(learning_rate, gradients[2])
                self.output_b.update(learning_rate, gradients[3])

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
        self.batch_size = 25             # 10
        self.hidden_layer_size = 350     # 350
        self.num_labels = 10

        # hidden layer 1
        self.w_1 = nn.Parameter(784, self.hidden_layer_size)
        self.b_1 = nn.Parameter(1, self.hidden_layer_size)

        # hidden layer 2
        self.w_2 = nn.Parameter(self.hidden_layer_size, self.hidden_layer_size)
        self.b_2 = nn.Parameter(1, self.hidden_layer_size)

        # hidden layer 3
        self.w_3 = nn.Parameter(self.hidden_layer_size, self.hidden_layer_size)
        self.b_3 = nn.Parameter(1, self.hidden_layer_size)

        # output vector
        self.output_wt = nn.Parameter(self.hidden_layer_size, self.num_labels)
        self.output_bias = nn.Parameter(1, self.num_labels)

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
        trans_1 = nn.Linear(x, self.w_1)
        trans_bias_1 = nn.AddBias(trans_1, self.b_1)
        layer_1 = nn.ReLU(trans_bias_1)

        # hidden layer 2
        trans_2 = nn.Linear(layer_1, self.w_2)
        trans_bias_2 = nn.AddBias(trans_2, self.b_2)
        layer_2 = nn.ReLU(trans_bias_2)

        # hidden layer 3
        trans_3 = nn.Linear(layer_2, self.w_3)
        trans_bias_3 = nn.AddBias(trans_3, self.b_3)
        layer_3 = nn.ReLU(trans_bias_3)

        # output vector (no relu)
        last_trans = nn.Linear(layer_3, self.output_wt)
        return nn.AddBias(last_trans, self.output_bias)


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
        y_hats = self.run(x)
        return nn.SoftmaxLoss(y_hats, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        adjusted_rate = -0.12
        done_training = 0

        while done_training < 0.98:

            for row_vect, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(row_vect, y)
                params = ([self.w_1, self.w_2, self.w_3, self.output_wt,
                           self.b_1, self.b_2, self.b_3, self.output_bias])
                gradients = nn.gradients(params, loss) # (loss, params)
                learning_rate = min(-0.005, adjusted_rate)

                # updates
                self.w_1.update(learning_rate, gradients[0])
                self.w_2.update(learning_rate, gradients[1])
                self.w_3.update(learning_rate, gradients[2])
                self.output_wt.update(learning_rate, gradients[3])
                self.b_1.update(learning_rate, gradients[4])
                self.b_2.update(learning_rate, gradients[5])
                self.b_3.update(learning_rate, gradients[6])
                self.output_bias.update(learning_rate, gradients[7])

            adjusted_rate += 0.05
            # check for 98 % accuracy after each epoch, not after each batch
            done_training = dataset.get_validation_accuracy()

