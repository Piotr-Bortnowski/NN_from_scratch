import numpy as np


# Helper functions

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

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
    return 2 * (y_pred - y_true) / y_true.shape[0] # or y_true.size idk

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy(y_true, y_pred):
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

def cross_entropy_derivative(y_true, y_pred):
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -(y_true / y_pred) / y_true.shape[0]

def softmax_cross_entropy_derivative(y_true, y_pred):
    return (y_pred - y_true) / y_true.shape[0]


def train_val_test_split(X, y, val_ratio=0.15, test_ratio=0.15, shuffle=True):
    n_samples = X.shape[0]
    if shuffle:
        indices = np.random.permutation(n_samples)
        X = X[indices]
        y = y[indices]

    val_size = int(n_samples * val_ratio)
    test_size = int(n_samples * test_ratio)

    # Get splits from sizes
    X_test = X[: test_size]
    y_test = y[: test_size]

    X_val = X[test_size: test_size + val_size]
    y_val = y[test_size: test_size + val_size]

    X_train = X[test_size + val_size:]
    y_train = y[test_size + val_size:]

    return X_train, y_train, X_val, y_val, X_test, y_test




# Dictionary for mapping activation function -> method and its derivative
# stored in a tuple (fun, fun_der)
activation_funcs_dict = {
        "sigmoid": (sigmoid, sigmoid_derivative),
        "relu": (relu, relu_derivative),
        "leaky_relu": (leaky_relu, leaky_relu_derivative),
        "softmax": (softmax, None)
    }

loss_funcs_dict = {
    "MSE": (MSE, MSE_derivative),
    "CrossEntropy": (cross_entropy,)
}

class MyNN:

    def __init__(self, architecture, activations, loss_func,
                 learning_rate=0.001, moment_decay_1=0.9, moment_decay_2=0.999, L2_reg_coef=0.001, dropout_rate=0.2):
        if len(architecture) != len(activations) + 1:
            raise ValueError("There must be exactly one more layer than activation functions")

        self.num_weight_layers = len(architecture) - 1
        self.architecture = np.array(architecture)
        self.lr = learning_rate
        self.B_1 = moment_decay_1
        self.B_2 = moment_decay_2
        self.L2_coef = L2_reg_coef
        self.dropout_rate = dropout_rate

        # Setting activation funcs and derivatives
        self.activations_funcs = [activation_funcs_dict[fun][0] for fun in activations]
        self.activation_func_derivatives = [activation_funcs_dict[fun][1] for fun in activations]

        # Setting loss function
        self.loss_func = loss_funcs_dict[loss_func][0]
        self.loss_func_derivative = loss_funcs_dict[loss_func][1]

        # Lists for saving values for backprop calculated in forward pass
        self.A = []
        self.Z = []
        self.dropout_masks = []

        # Initializing weights based on layers and biases
        self.weights = []
        self.biases = []

        # Xavier/Glorot Initialization
        for i in range(self.num_weight_layers):
            in_dim = self.architecture[i]
            out_dim = self.architecture[i + 1]
            limit = np.sqrt(2.0 / (in_dim + out_dim))
            self.weights.append(np.random.randn(in_dim, out_dim) * limit)
            self.biases.append(np.zeros((1, out_dim)))

        # First moment
        self.m_W = [np.zeros_like(W) for W in self.weights]
        self.m_b = [np.zeros_like(b) for b in self.biases]

        # Second moment
        self.v_W = [np.zeros_like(W) for W in self.weights]
        self.v_b = [np.zeros_like(b) for b in self.biases]

    def forward(self, x, training=False):
        self.A = []
        self.Z = []
        self.dropout_masks = []


        self.A.append(x)
        for i in range(self.num_weight_layers):
            x = np.dot(x, self.weights[i]) + self.biases[i]
            self.Z.append(x)
            x = self.activations_funcs[i](x)

            is_hidden_layer = i < (self.num_weight_layers - 1)
            if training and is_hidden_layer and self.dropout_rate > 0:
                keep_prob = 1 - self.dropout_rate
                mask = np.random.rand(*x.shape) < keep_prob
                x = (x * mask) / keep_prob
                self.dropout_masks.append(mask)
            else:
                self.dropout_masks.append(None)

            self.A.append(x)
        return x

    def backward(self, y_true, t):
        n_batch = y_true.shape[0] # samples in batch

        if self.loss_func == cross_entropy and self.activations_funcs[-1] == softmax:
            dZ_last = (self.A[-1] - y_true) / n_batch
        else:
            dA_last = self.loss_func_derivative(y_true, self.A[-1])

            if self.activations_funcs[-1] == softmax:
                # Softmax with any other loss function (e.g. MSE)
                A_last = self.A[-1]
                sum_dA_A = np.sum(dA_last * A_last, axis=1, keepdims=True)
                dZ_last = A_last * (dA_last - sum_dA_A)
            else:
                # Standard activation functions (Sigmoid, ReLU, LeakyReLU)
                dZ_last = dA_last * self.activation_func_derivatives[-1](self.Z[-1]) # shape (N_batch_size X D_out)


        eps = 10 ** -8
        for i in range(self.num_weight_layers - 1, -1, -1):

            # calculate partial derivatives of weights and biases with respect to cost
            dW = np.dot(self.A[i].T, dZ_last)   #(NxIN).T * NxOUT
                                                # INxN * NxOUT => shape of weights
            dB = np.sum(dZ_last, axis=0, keepdims=True) # summing bias gradients over a batch

            if self.L2_coef > 0:  # add gradient for L2 regularization
                dW += (self.L2_coef / n_batch) * self.weights[i]

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
                if self.dropout_masks[i - 1] is not None:
                    dA_last = dA_last * self.dropout_masks[i - 1] / (1 - self.dropout_rate)
                dZ_last = dA_last * self.activation_func_derivatives[i - 1](self.Z[i-1])

            self.weights[i] -= step_W
            self.biases[i] -= step_b


    def train(self, X, y, epochs, batch_size=32, shuffle=True, val_data=None):
        num_samples = X.shape[0]
        if batch_size == None or batch_size >= X.shape[0]:
            batch_size = num_samples

        t = 0
        EPOCHS_TO_PRINT = 10

        for epoch in range(epochs):
            # shuffle samples
            if shuffle:
                indices = np.random.permutation(num_samples)
                X_shuffled = X[indices]
                y_shuffled = y[indices]
            else:
                X_shuffled = X
                y_shuffled = y

            train_loss = 0.0
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
                y_pred = self.forward(current_batch_X, training=True)
                self.backward(current_batch_y, t)

                # calculating loss
                batch_loss = self.compute_loss(current_batch_y, y_pred)
                train_loss += batch_loss * (batch_end - batch_start)



            train_loss /= num_samples
            # calculating validation loss
            if val_data is not None:
                X_val, y_val = val_data

                y_val_pred = self.predict(X_val)
                val_loss = self.compute_loss(y_val, y_val_pred)

            if (epoch + 1) % EPOCHS_TO_PRINT == 0 or epoch == 0:
                if (epoch + 1) % EPOCHS_TO_PRINT == 0 or epoch == 0:
                    if val_data is None:
                        print(f"Epoch {epoch + 1}/{epochs} - Train loss: {train_loss:.4f}")
                    else:
                        print( f"Epoch {epoch + 1}/{epochs} - Train loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}")


    def predict(self, X):
        return self.forward(X, training=False)

    def compute_loss(self, y_true, y_pred):
        # Chosen loss function
        data_loss = self.loss_func(y_true, y_pred)

        # L2 penalty - cost of all weights squared
        if self.L2_coef > 0:
            n = y_true.shape[0]
            l2_cost = sum(np.sum(W ** 2) for W in self.weights)
            return data_loss + (self.L2_coef / (2 * n)) * l2_cost
        return data_loss

if __name__ == "__main__":
    np.random.seed(42)

    # Random dataset
    X_data = np.random.uniform(-2, 2, (1000, 2))
    y_data = ((X_data[:, 0] ** 2 + X_data[:, 1] ** 2) < 1.0).astype(int).reshape(-1, 1)

    # Split the dataset
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(
        X=X_data, y=y_data, val_ratio=0.15, test_ratio=0.15
    )

    nn = MyNN(
        architecture=[2, 16, 8, 1],
        activations=["leaky_relu", "leaky_relu", "sigmoid"],
        loss_func="MSE",
        learning_rate=0.01,
        dropout_rate=0.1,
        L2_reg_coef=0.001
    )


    nn.train(X_train, y_train, epochs=100, batch_size=32, val_data=(X_val, y_val))
