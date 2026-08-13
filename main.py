import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def relu(x):
    return np.where(x > 0, x, 0)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, x * alpha)

def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)

def MSE(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def MSE_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.shape[0]

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy(y_true, y_pred):
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

def softmax_cross_entropy_derivative(y_true, y_pred):
    return (y_pred - y_true) / y_true.shape[0]

# Dictionary for mapping activation function -> method and its derivative
# stored in a tuple (fun, fun_der)
activation_funcs_dict = {
        "sigmoid": (sigmoid, sigmoid_derivative),
        "relu": (relu, relu_derivative),
        "leaky_relu": (leaky_relu, leaky_relu_derivative),
        "softmax": (softmax, softmax_cross_entropy_derivative)
    }

loss_funcs_dict = {
    "MSE": (MSE, MSE_derivative),
    "CrossEntropy": (cross_entropy, softmax_cross_entropy_derivative)
}

class MyNN:

    def __init__(self, architecture, activations, loss_func,
                 learning_rate=0.001, moment_decay_1=0.9, moment_decay_2=0.999, L2_reg_coef=0.001):
        if len(architecture) != len(activations) + 1:
            raise ValueError("There must be exactly one more layer than activation functions")

        self.num_weight_layers = len(architecture) - 1
        self.architecture = np.array(architecture)
        self.lr = learning_rate
        self.B_1 = moment_decay_1
        self.B_2 = moment_decay_2
        self.L2_coef = L2_reg_coef

        # Setting activation funcs and derivatives
        self.activations_funcs = [activation_funcs_dict[fun][0] for fun in activations]
        self.activation_func_derivatives = [activation_funcs_dict[fun][1] for fun in activations]

        # Setting loss function
        self.loss_func = loss_funcs_dict[loss_func][0]
        self.loss_func_derivative = loss_funcs_dict[loss_func][1]

        # Lists for saving values for backprop calculated in forward pass
        self.A = []
        self.Z = []

        # Initializing weights based on layers and biases
        self.weights = []
        self.biases = []
        for i in range(self.num_weight_layers):
            # Weights with "He Normal initialization"
            in_dim = self.architecture[i]
            out_dim = self.architecture[i + 1]
            self.weights.append(np.sqrt(2/in_dim) * np.random.randn(in_dim, out_dim))
            # Biases
            self.biases.append(np.zeros((1, out_dim)))

        # First moment
        self.m_W = [np.zeros_like(W) for W in self.weights]
        self.m_b = [np.zeros_like(b) for b in self.biases]

        # Second moment
        self.v_W = [np.zeros_like(W) for W in self.weights]
        self.v_b = [np.zeros_like(b) for b in self.biases]

    def forward(self, x):
        self.A = []
        self.Z = []
        self.A.append(x)
        for i in range(self.num_weight_layers):
            x = np.dot(x, self.weights[i]) + self.biases[i]
            self.Z.append(x)
            x = self.activations_funcs[i](x)
            self.A.append(x)
        return x

    def backward(self, y_true, t):
        if self.loss_func == cross_entropy and self.activations_funcs[-1] == softmax:
            dZ_last = self.loss_func_derivative(y_true, self.A[-1])
        else:
            dA_last = self.loss_func_derivative(y_true, self.A[-1])
            dZ_last = dA_last * self.activation_func_derivatives[-1](self.Z[-1]) # shape (N_batch_size X D_out)


        eps = 10 ** -8
        for i in range(self.num_weight_layers - 1, -1, -1):

            # calculate partial derivatives of weights and biases with respect to cost
            dW = np.dot(self.A[i].T, dZ_last)   #(NxIN).T * NxOUT
                                                # INxN * NxOUT => shape of weights
            dB = np.sum(dZ_last, axis=0, keepdims=True) # summing bias gradients over a batch

            self.m_W[i] = self.B_1 * self.m_W[i] + (1 - self.B_1) * dW
            self.v_W[i] = self.B_2 * self.v_W[i] + (1 - self.B_2) * (dW ** 2)

            m_hat_W = self.m_W[i] / (1 - self.B_1 ** t)
            v_hat_W = self.v_W[i] / (1 - self.B_2 ** t)

            step_W = self.lr * m_hat_W / (np.sqrt(v_hat_W) + eps)

            self.m_b[i] = self.B_1 * self.m_b[i] + (1 - self.B_1) * dB
            self.v_b[i] = self.B_2 * self.v_b[i] + (1 - self.B_2) * (dB ** 2)

            m_hat_b = self.m_b[i] / (1 - self.B_1 ** t)
            v_hat_b = self.v_b[i] / (1 - self.B_2 ** t)

            step_b = self.lr * m_hat_b / (np.sqrt(v_hat_b) + eps)


            if i > 0:
                dA_last = np.dot(dZ_last, self.weights[i].T)
                dZ_last = dA_last * self.activation_func_derivatives[i - 1](self.Z[i-1])

            n = y_true.shape[0]  # samples in batch

            self.weights[i] -= (step_W + self.lr * (self.L2_coef / n) * self.weights[i])  # add gradient for L2 regularization
            self.biases[i] -= step_b



    def train(self, X, y, epochs, batch_size=32, shuffle=True):
        num_samples = X.shape[0]
        if batch_size == None or batch_size >= X.shape[0]:
            batch_size = num_samples

        t = 0
        EPOCHS_TO_PRINT = 100

        for epoch in range(epochs):
            # shuffle samples
            if shuffle:
                indices = np.random.permutation(num_samples)
                X_shuffled = X[indices]
                y_shuffled = y[indices]
            else:
                X_shuffled = X
                y_shuffled = y

            current_epoch_loss = 0.0
            # iterate over all batches
            num_batches = int(np.ceil(num_samples / batch_size))
            for i in range(num_batches): # number of batches
                # batch indexes
                batch_start = i * batch_size
                batch_end = min(batch_start + batch_size, num_samples)
                current_batch_X = X_shuffled[batch_start:batch_end]
                current_batch_y = y_shuffled[batch_start:batch_end]

                t += 1
                # forward and backprop
                y_pred = self.forward(current_batch_X)
                self.backward(current_batch_y, t)
                # calculating loss
                if epoch % EPOCHS_TO_PRINT == 0:
                    batch_n = current_batch_y.shape[0]
                    batch_loss = self.loss_func(current_batch_y, y_pred)
                    current_epoch_loss += batch_loss * batch_n

            if epoch % EPOCHS_TO_PRINT == 0:
                current_epoch_loss /= num_samples

                l2_penalty = sum(np.sum(W ** 2) for W in self.weights)
                l2_cost = (self.L2_coef * l2_penalty) / (2 * num_samples)

                current_epoch_loss += l2_cost

                print(f"Epoch {epoch} - Loss: {current_epoch_loss}")



nn = MyNN([2, 4, 1], ["leaky_relu", "sigmoid"], "MSE", learning_rate=0.01, )

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])
nn.train(X, y, 10000)
preds = nn.forward(X)
print(preds)

