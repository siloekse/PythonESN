import numpy as np

from scipy import sparse
from scipy.io import loadmat
from scipy.sparse import linalg as slinalg

from sklearn.decomposition import KernelPCA, PCA
from sklearn.linear_model import ElasticNet, Lasso, Ridge, LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVR

def NRMSE(y_true, y_pred):
    """ Normalized Root Mean Squared Error """
    y_std = np.std(y_true)

    return np.sqrt(mean_squared_error(y_true, y_pred))/y_std

class ESN(object):
    def __init__(self, n_internal_units = 100, spectral_radius = 0.9, connectivity = 0.5, input_scaling = 0.5, input_shift = 0.0, teacher_scaling = 0.5, teacher_shift = 0.0, feedback_scaling = 0.01, noise_level = 0.01):
        # Initialize attributes
        self._n_internal_units = n_internal_units
        self._spectral_radius = spectral_radius
        self._connectivity = connectivity

        self._input_scaling = input_scaling
        self._input_shift = input_shift
        self._teacher_scaling = teacher_scaling
        self._teacher_shift = teacher_shift
        self._feedback_scaling = feedback_scaling
        self._noise_level = noise_level
        self._dim_output = None

        # The weights will be set later, when data is provided
        self._input_weights = None
        self._feedback_weights = None

        # Regression method and embedding method.
        # Initialized to None for now. Will be set during 'fit'.
        self._regression_method = None
        self._embedding_method = None

        # Generate internal weights
        self._internal_weights = self._initialize_internal_weights(n_internal_units, connectivity, spectral_radius)

    def fit(self, Xtr, Ytr, n_drop = 100, regression_method = 'linear', regression_parameters = None, embedding = 'identity', n_dim = 3, embedding_parameters = None):
        _,_ = self._fit_transform(Xtr = Xtr, Ytr = Ytr, n_drop = n_drop, regression_method = regression_method, regression_parameters = regression_parameters, embedding = embedding, n_dim = n_dim, embedding_parameters = embedding_parameters)

        return

    def _fit_transform(self, Xtr, Ytr, n_drop = 100, regression_method = 'linear', regression_parameters = None, embedding = 'identity', n_dim = 3, embedding_parameters = None):
        n_data, dim_data = Xtr.shape
        _, dim_output = Ytr.shape

        self._dim_output = dim_output

        # If this is the first time the network is tuned, set the input and feedback weights.
        # The weights are dense and uniformly distributed in [-1.0, 1.0]
        if (self._input_weights is None):
            self._input_weights = 2.0*np.random.rand(self._n_internal_units, dim_data) - 1.0

        if (self._feedback_weights is None):
            self._feedback_weights = 2.0*np.random.rand(self._n_internal_units, dim_output) - 1.0

        # Initialize regression method
        if (regression_method == 'nusvr'):
            # NuSVR, RBF kernel
            C, nu, gamma = regression_parameters
            self._regression_method = NuSVR(C = C, nu = nu, gamma = gamma)

        elif (regression_method == 'linsvr'):
            # NuSVR, linear kernel
            C = regression_parameters[0]
            nu = regression_parameters[1]

            self._regression_method = NuSVR(C = C, nu = nu, kernel='linear')

        elif (regression_method == 'enet'):
            # Elastic net
            alpha, l1_ratio = regression_parameters
            self._regression_method = ElasticNet(alpha = alpha, l1_ratio = l1_ratio)

        elif (regression_method == 'ridge'):
            # Ridge regression
            self._regression_method = Ridge(alpha = regression_parameters)

        elif (regression_method == 'lasso'):
            # LASSO
            self._regression_method = Lasso(alpha = regression_parameters)

        else:
            # Use canonical linear regression
            self._regression_method = LinearRegression()

        # Initialize embedding method
        if (embedding == 'identity'):
            self._embedding_dimensions = self._n_internal_units
        else:
            self._embedding_dimensions = n_dim

            if (embedding == 'kpca'):
                # Kernel PCA with RBF kernel
                self._embedding_method = KernelPCA(n_components = n_dim, kernel = 'rbf', gamma = embedding_parameters)

            elif (embedding == 'pca'):
                # PCA
                self._embedding_method = PCA(n_components = n_dim)

            else:
                raise(ValueError, "Unknown embedding method")

        # Calculate states/embedded states.
        # Note: If the embedding is 'identity', embedded states will be equal to the states.
        states, embedded_states,_ = self._compute_state_matrix(X = Xtr, Y = Ytr, n_drop = n_drop)

        # Train output
        self._regression_method.fit(np.concatenate((embedded_states, self._scaleshift(Xtr[n_drop:,:], self._input_scaling, self._input_shift)), axis=1),
                self._scaleshift(Ytr[n_drop:,:], self._teacher_scaling, self._teacher_shift).flatten())

        return states, embedded_states

    def predict(self, X, Y = None, n_drop = 100, error_function = NRMSE):
        Yhat, error, _, _ = self._predict_transform(X = X, Y = Y, n_drop = n_drop, error_function = error_function)

        return Yhat, error

    def _predict_transform(self, X, Y = None, n_drop = 100, error_function = NRMSE):
        # Predict outputs
        states,embedded_states,Yhat = self._compute_state_matrix(X = X, n_drop = n_drop)

        # Revert scale and shift
        Yhat = self._uscaleshift(Yhat, self._teacher_scaling, self._teacher_shift)

        # Compute error if ground truth is provided
        if (Y is not None):
            error = error_function(Y[n_drop:,:], Yhat)

        return Yhat, error, states, embedded_states

    def _compute_state_matrix(self, X, Y = None, n_drop = 100):
        n_data, _ = X.shape

        # Initial values
        previous_state = np.zeros((1, self._n_internal_units), dtype=float)
        previous_output = np.zeros((1, self._dim_output), dtype=float)

        # Storage
        state_matrix = np.empty((n_data - n_drop, self._n_internal_units), dtype=float)
        embedded_states = np.empty((n_data - n_drop, self._embedding_dimensions), dtype=float)
        outputs = np.empty((n_data - n_drop, self._dim_output), dtype=float)

        for i in range(n_data):
            # Process inputs
            previous_state = np.atleast_2d(previous_state)
            current_input = np.atleast_2d(self._scaleshift(X[i, :], self._input_scaling, self._input_shift))
            feedback = self._feedback_scaling*np.atleast_2d(previous_output)

            # Calculate state. Add noise and apply nonlinearity.
            state_before_tanh = self._internal_weights.dot(previous_state.T) + self._input_weights.dot(current_input.T) + self._feedback_weights.dot(feedback.T)
            state_before_tanh += np.random.rand(self._n_internal_units, 1)*self._noise_level
            previous_state = np.tanh(state_before_tanh).T

            # Embed data and perform regression if applicable.
            if (Y is not None):
                # If we are training, the previous output should be a scaled and shifted version of the ground truth.
                previous_output = self._scaleshift(Y[i, :], self._teacher_scaling, self._teacher_shift)
            else:
                # Should the data be embedded?
                if (self._embedding_method is not None):
                    current_embedding = self._embedding_method.transform(previous_state)
                else:
                    current_embedding = previous_state

                # Perform regression
                previous_output = self._regression_method.predict(np.concatenate((current_embedding, current_input), axis=1))

            # Store everything after the dropout period
            if (i > n_drop - 1):
                state_matrix[i - n_drop, :] = previous_state.flatten()

                # Only save embedding for test data.
                # In training, we do it after computing the whole state matrix.
                if (Y is None):
                    embedded_states[i - n_drop, :] = current_embedding.flatten()

                outputs[i - n_drop, :] = previous_output.flatten()

        # Now, embed the data if we are in training
        if (Y is not None):
            if (self._embedding_method is not None):
                embedded_states = self._embedding_method.fit_transform(state_matrix)
            else:
                embedded_states = state_matrix

        return state_matrix, embedded_states, outputs

    def _scaleshift(self, x, scale, shift):
        # Scales and shifts x by scale and shift
        return (x*scale + shift)

    def _uscaleshift(self, x, scale, shift):
        # Reverts the scale and shift applied by _scaleshift
        return ( (x - shift)/float(scale) )

    def _initialize_internal_weights(self, n_internal_units, connectivity, spectral_radius):
        # The eigs function might not converge. Attempt until it does.
        convergence = False
        while (not convergence):
            # Generate sparse, uniformly distributed weights.
            internal_weights = sparse.rand(n_internal_units, n_internal_units, density=connectivity).todense()

            # Ensure that the nonzero values are uniformly distributed in [-0.5, 0.5]
            internal_weights[np.where(internal_weights > 0)] -= 0.5

            try:
                # Get the largest eigenvalue
                w,_ = slinalg.eigs(internal_weights, k=1, which='LM')

                convergence = True

            except:
                continue

        # Adjust the spectral radius.
        internal_weights /= np.abs(w)/spectral_radius

        return internal_weights

def run_from_config(Xtr, Ytr, Xte, Yte, config):
    # Instantiate ESN object
    esn = ESN(n_internal_units = config['n_internal_units'],
              spectral_radius = config['spectral_radius'],
              connectivity = config['connectivity'],
              input_scaling = config['input_scaling'],
              input_shift = config['input_shift'],
              teacher_scaling = config['teacher_scaling'],
              teacher_shift = config['teacher_shift'],
              feedback_scaling = config['feedback_scaling'],
              noise_level = config['noise_level'])

    # Get parameters
    n_drop = config['n_drop']
    regression_method = config['regression_method']
    regression_parameters = config['regression_parameters']
    embedding = config['embedding']
    n_dim = config['n_dim']
    embedding_parameters = config['embedding_parameters']

    # Fit and predict
    esn.fit(Xtr, Ytr, n_drop = n_drop, regression_method = regression_method, regression_parameters = regression_parameters,
            embedding = embedding, n_dim = n_dim, embedding_parameters = embedding_parameters)

    Yhat,error = esn.predict(Xte, Yte)

    return Yhat, error

def run_from_config_return_states(Xtr, Ytr, Xte, Yte, config):
    # Instantiate ESN object
    esn = ESN(n_internal_units = config['n_internal_units'],
              spectral_radius = config['spectral_radius'],
              connectivity = config['connectivity'],
              input_scaling = config['input_scaling'],
              input_shift = config['input_shift'],
              teacher_scaling = config['teacher_scaling'],
              teacher_shift = config['teacher_shift'],
              feedback_scaling = config['feedback_scaling'],
              noise_level = config['noise_level'])

    # Get parameters
    n_drop = config['n_drop']
    regression_method = config['regression_method']
    regression_parameters = config['regression_parameters']
    embedding = config['embedding']
    n_dim = config['n_dim']
    embedding_parameters = config['embedding_parameters']

    # Fit and predict
    train_states, train_embedding = esn._fit_transform(Xtr, Ytr, n_drop = n_drop, regression_method = regression_method, regression_parameters = regression_parameters,
            embedding = embedding, n_dim = n_dim, embedding_parameters = embedding_parameters)

    Yhat, error, test_states, test_embedding = esn._predict_transform(Xte, Yte)

    return Yhat, error, train_states, train_embedding, test_states, test_embedding

def format_config(n_internal_units, spectral_radius, connectivity, input_scaling, input_shift, teacher_scaling, teacher_shift, feedback_scaling, noise_level,
        n_drop, regression_method, regression_parameters, embedding, n_dim, embedding_parameters):

    config = dict(
                n_internal_units = n_internal_units,
                spectral_radius = spectral_radius,
                connectivity = connectivity,
                input_scaling = input_scaling,
                input_shift = input_shift,
                teacher_scaling = teacher_scaling,
                teacher_shift = teacher_shift,
                feedback_scaling = feedback_scaling,
                noise_level = noise_level,
                n_drop = n_drop,
                regression_method = regression_method,
                regression_parameters = regression_parameters,
                embedding = embedding,
                n_dim = n_dim,
                embedding_parameters = embedding_parameters
            )

    return config

def generate_datasets(X, Y, test_percent = 0.15, val_percent = 0.15, scaler = StandardScaler):
    n_data,_ = X.shape

    n_te = np.ceil(test_percent*n_data).astype(int)
    n_val = np.ceil(val_percent*n_data).astype(int)
    n_tr = n_data - n_te - n_val

    # Split dataset
    Xtr = X[:n_tr, :]
    Ytr = Y[:n_tr, :]

    Xval = X[n_tr:-n_te, :]
    Yval = Y[n_tr:-n_te, :]

    Xte = X[-n_te:, :]
    Yte = Y[-n_te:, :]

    # Scale
    Xscaler = scaler()
    Yscaler = scaler()

    # Fit scaler on training set
    Xtr = Xscaler.fit_transform(Xtr)
    Ytr = Yscaler.fit_transform(Ytr)

    # Transform the rest
    Xval = Xscaler.transform(Xval)
    Yval = Yscaler.transform(Yval)

    Xte = Xscaler.transform(Xte)
    Yte = Yscaler.transform(Yte)

    return Xtr, Ytr, Xval, Yval, Xte, Yte

def construct_output(X, shift):
    return X[:-shift,:], X[shift:, :]

def load_lorenz(path, shift):
    data = loadmat(path)

    X, Y = construct_output(data['X'], shift)

    return X, Y

def load_from_text(path):
    data = np.loadtxt(path)

    return np.atleast_2d(data[:, 0]).T, np.atleast_2d(data[:, 1]).T

def load_from_dir(path):
    Xtr_base = np.loadtxt(path + '/Xtr')
    Ytr_base = np.loadtxt(path + '/Ytr')
    Xval_base = np.loadtxt(path + '/Xval')
    Yval_base = np.loadtxt(path + '/Yval')
    Xte_base = np.loadtxt(path + '/Xte')
    Yte_base = np.loadtxt(path + '/Yte')

    Xtr, Ytr, Xval, Yval, Xte, Yte = np.atleast_2d(Xtr_base, Ytr_base, Xval_base, Yval_base, Xte_base, Yte_base)

    # Need axis 0 to be the samples
    if Xtr.shape[0] == 1:
        Xtr = Xtr.T
    if Ytr.shape[0] == 1:
        Ytr = Ytr.T

    if Xval.shape[0] == 1:
        Xval = Xval.T
    if Yval.shape[0] == 1:
        Yval = Yval.T

    if Xte.shape[0] == 1:
        Xte = Xte.T
    if Yte.shape[0] == 1:
        Yte = Yte.T

    return Xtr, Ytr, Xval, Yval, Xte, Yte

if __name__ == "__main__":
    pass
