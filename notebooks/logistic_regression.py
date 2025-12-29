import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def map_feature(X1, X2, degree=6):
    """
    Feature mapping function to polynomial features (OPTIMIZED VERSION).
    
    Parameters:
    X1: array-like, first feature
    X2: array-like, second feature
    degree: int, degree of polynomial features (default 6)
    
    Returns:
    out: array, polynomial features
    """
    X1 = np.atleast_1d(X1)
    X2 = np.atleast_1d(X2)
    
    # Pre-allocate list for better performance
    out = []
    
    # Generate polynomial features using vectorized operations
    for i in range(1, degree+1):
        for j in range(i + 1):
            # Vectorized computation: X1**(i-j) * X2**j for all elements at once
            out.append((X1**(i-j) * (X2**j)))
    
    # Stack all features into a single array (more efficient than appending)
    return np.stack(out, axis=1)

# UNQ_C1
# GRADED CELL: my_softmax
def my_softmax(z):
    """ Softmax converts a vector of values to a probability distribution.
    Handles both 1D and 2D arrays (applies softmax row-wise for 2D).
    Args:
      z (ndarray (N,) or (m, K))  : input data, N features or m examples with K classes
    Returns:
      a (ndarray (N,) or (m, K))  : softmax of z
    """
    # Handle 2D arrays (apply softmax row-wise)
    if z.ndim == 2:
        # For numerical stability, subtract max from each row
        z_shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    # Handle 1D arrays (original implementation)
    ### START CODE HERE ###
    N = len(z)
    a = np.zeros(N)
    e_sum = 0
    
    for k in range(N):
        e_sum += np.exp(z[k])
    for j in range(N):
        a[j] = np.exp(z[j])/e_sum
    
    ### END CODE HERE ###
    return a
    
class LogisticRegression_lr:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations

    def sigmoid(self, z):
        """
        Compute the sigmoid function.
        
        Parameters:
        z: array-like, input values
        
        Returns:
        sigmoid values
        """
        # Clip z to prevent numerical overflow when computing exp(-z)
        # For very negative z (e.g., z < -500), exp(-z) = exp(|z|) becomes extremely large
        # and can overflow to inf, causing numerical instability.
        # For z <= -500, sigmoid(z) ≈ 0.0 (within machine precision)
        # For z >= 500, sigmoid(z) ≈ 1.0 (within machine precision)
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def compute_cost(self, X, y, w, b, lambda_= 1):
        """
        Computes the cost over all examples (VECTORIZED VERSION)
        Args:
        X : (ndarray Shape (m,n)) data, m examples by n features
        y : (array_like Shape (m,)) target value 
        w : (array_like Shape (n,)) Values of parameters of the model      
        b : scalar Values of bias parameter of the model
        lambda_: unused placeholder
        Returns:
        total_cost: (scalar)         cost 
        """
        m = X.shape[0]
        
        # Vectorized computation: z = X @ w + b for all examples at once
        z = X @ w + b
        
        # Vectorized sigmoid computation
        f_wb = self.sigmoid(z)
        
        # Vectorized cost computation using element-wise operations
        # Avoid log(0) by using np.clip or adding small epsilon
        epsilon = 1e-15
        f_wb = np.clip(f_wb, epsilon, 1 - epsilon)
        
        # Binary cross-entropy loss: -y*log(f) - (1-y)*log(1-f)
        loss = -y * np.log(f_wb) - (1 - y) * np.log(1 - f_wb)
        
        # Average loss
        total_cost = np.mean(loss)
        
        return total_cost

    def compute_gradient(self, X, y, w, b, lambda_=None): 
        """
        Computes the gradient for logistic regression (VECTORIZED VERSION)
    
        Args:
        X : (ndarray Shape (m,n)) variable such as house size 
        y : (array_like Shape (m,1)) actual value 
        w : (array_like Shape (n,1)) values of parameters of the model      
        b : (scalar)                 value of parameter of the model 
        lambda_: unused placeholder.
        Returns
        dj_dw: (array_like Shape (n,1)) The gradient of the cost w.r.t. the parameters w. 
        dj_db: (scalar)                The gradient of the cost w.r.t. the parameter b. 
        """
        m = X.shape[0]
        
        # Vectorized computation: z = X @ w + b for all examples at once
        z = X @ w + b
        
        # Vectorized sigmoid computation
        f_wb = self.sigmoid(z)
        
        # Vectorized error computation: error = f_wb - y for all examples
        err = f_wb - y
        
        # Vectorized gradient computation
        # dj_db = mean(err) = mean(f_wb - y)
        dj_db = np.mean(err)
        
        # dj_dw = mean(err * X) = mean((f_wb - y) * X) for each feature
        # This is equivalent to: X.T @ err / m
        dj_dw = (X.T @ err) / m
        
        return dj_db, dj_dw
    
    def compute_cost_regularized(self, X, y, w, b, lambda_ = 1):
        """
        Computes the cost over all examples (VECTORIZED VERSION)
        Args:
        X : (array_like Shape (m,n)) data, m examples by n features
        y : (array_like Shape (m,)) target value 
        w : (array_like Shape (n,)) Values of parameters of the model      
        b : (scalar) Values of bias parameter of the model
        lambda_ : (scalar, float)    Controls amount of regularization
        Returns:
        total_cost: (scalar)         cost 
        """
        m = X.shape[0]
        
        # Calls the compute_cost function that you implemented above
        cost_without_reg = self.compute_cost(X, y, w, b) 
        
        # Vectorized regularization cost: sum of squares of w
        # reg_cost = sum(w^2) = w.T @ w = np.sum(w**2)
        reg_cost = np.sum(w**2)
        
        # Add the regularization cost to get the total cost
        total_cost = cost_without_reg + (lambda_/(2 * m)) * reg_cost

        return total_cost

    def compute_gradient_regularized(self, X, y, w, b, lambda_ = 1): 
        """
        Computes the gradient for linear regression (VECTORIZED VERSION)
    
        Args:
        X : (ndarray Shape (m,n))   variable such as house size 
        y : (ndarray Shape (m,))    actual value 
        w : (ndarray Shape (n,))    values of parameters of the model      
        b : (scalar)                value of parameter of the model  
        lambda_ : (scalar,float)    regularization constant
        Returns
        dj_db: (scalar)             The gradient of the cost w.r.t. the parameter b. 
        dj_dw: (ndarray Shape (n,)) The gradient of the cost w.r.t. the parameters w. 

        """
        m = X.shape[0]
        
        dj_db, dj_dw = self.compute_gradient(X, y, w, b)

        # Vectorized regularization term: (lambda_ / m) * w
        # This adds regularization to all elements of dj_dw at once
        dj_dw = dj_dw + (lambda_ / m) * w
            
        return dj_db, dj_dw

    def gradient_descent(self, X, y, w_in, b_in, alpha, num_iters, lambda_, regularized=False): 
        """
        Performs batch gradient descent to learn theta. Updates theta by taking 
        num_iters gradient steps with learning rate alpha
        
        Args:
        X :    (array_like Shape (m, n)
        y :    (array_like Shape (m,))
        w_in : (array_like Shape (n,))  Initial values of parameters of the model
        b_in : (scalar)                 Initial value of parameter of the model
        cost_function:                  function to compute cost
        alpha : (float)                 Learning rate
        num_iters : (int)               number of iterations to run gradient descent
        lambda_ (scalar, float)         regularization constant
        
        Returns:
        w : (array_like Shape (n,)) Updated values of parameters of the model after
            running gradient descent
        b : (scalar)                Updated value of parameter of the model after
            running gradient descent
        """

        if regularized == False:
            cost_function = self.compute_cost
            gradient_function = self.compute_gradient
        else:
            cost_function = self.compute_cost_regularized
            gradient_function = self.compute_gradient_regularized

        
        # number of training examples
        m = len(X)
        
        # An array to store cost J and w's at each iteration primarily for graphing later
        J_history = []
        w_history = []
        dj_dw_history = []
        dj_db_history = []
        
        for i in range(num_iters):

            # Calculate the gradient and update the parameters
            dj_db, dj_dw = gradient_function(X, y, w_in, b_in, lambda_)

            # Add to history
            dj_dw_history.append(dj_dw)
            dj_db_history.append(dj_db)

            # Update Parameters using w, b, alpha and gradient
            w_in = w_in - alpha * dj_dw               
            b_in = b_in - alpha * dj_db              
        
            # Save cost J at each iteration
            if i<100000:      # prevent resource exhaustion 
                cost = cost_function(X, y, w_in, b_in, lambda_)
                J_history.append(cost)

            # Print cost every at intervals 10 times or as many iterations if < 10
            if i% np.ceil(num_iters/10) == 0 or i == (num_iters-1):
                w_history.append(w_in)
                print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.4f}   ")

        self.weights = w_in
        self.bias = b_in
            
        return dj_dw_history, dj_db_history, w_in, b_in, J_history, w_history #return w and J,w history for graphing

    def gradient_descent_softmax(self, X, Y_one_hot, initial_all_w, initial_all_b, alpha, num_iters, lambda_=0.0):
        """
        Performs batch gradient descent for Softmax Regression with optional L2 regularization.
        
        Args:
        X (ndarray (m, n)): Input features.
        Y_one_hot (ndarray (m, K)): True labels in one-hot format.
        initial_all_w (ndarray (K, n)): Initial weights matrix.
        initial_all_b (ndarray (K, 1)): Initial bias vector.
        alpha (float): Learning rate.
        num_iters (int): Number of iterations.
        lambda_ (float): Regularization parameter (default 0.0, no regularization).
        
        Returns:
        all_w (ndarray (K, n)): Final learned weights matrix.
        all_b (ndarray (K, 1)): Final learned bias vector.
        J_history (list): History of the cost function J.
        """
        
        m, n = X.shape
        K = initial_all_w.shape[0]  # Number of classes
        
        all_w = initial_all_w
        all_b = initial_all_b
        J_history = []
        
        for i in range(num_iters):
            
            # 1. Forward Propagation (Calculate Scores and Probabilities)
            
            # Calculate Scores (Logits): Z = X @ all_w.T + all_b.T (Broadcasting)
            # Z shape: (m, K)
            Z = X @ all_w.T + all_b.T
            
            # Calculate Softmax Probabilities: A (m, K)
            # Reusing the my_softmax function defined earlier
            A = my_softmax(Z) 
            
            # 2. Calculate Cost (Categorical Cross-Entropy with optional L2 regularization)
            # CCE Loss: J = -1/m * sum(Y_one_hot * log(A))
            # Note: Added a tiny epsilon to log(A) for numerical stability if needed
            cost = -np.sum(Y_one_hot * np.log(A + 1e-9)) / m
            
            # Add L2 regularization term if lambda_ > 0
            if lambda_ > 0:
                reg_cost = (lambda_ / (2 * m)) * np.sum(all_w ** 2)
                cost += reg_cost
            
            J_history.append(cost)

            # 3. Calculate Gradients (Backward Propagation)
            
            # Error / Gradient with respect to Scores: G = A - Y_one_hot
            # G shape: (m, K)
            G = A - Y_one_hot
            
            # Gradient for Weights (dW) - (K, n)
            # dW = 1/m * G.T @ X
            dW = (1 / m) * G.T @ X
            
            # Add L2 regularization gradient if lambda_ > 0
            if lambda_ > 0:
                dW = dW + (lambda_ / m) * all_w
            
            # Gradient for Biases (db) - (K, 1)
            # db = 1/m * sum(G along axis 0)
            db = (1 / m) * np.sum(G, axis=0, keepdims=True).T
            
            # 4. Update Parameters
            all_w = all_w - alpha * dW
            all_b = all_b - alpha * db
            
            # Print cost every at intervals 10 times or as many iterations if < 10
            if i% np.ceil(num_iters/10) == 0 or i == (num_iters-1):
                print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.4f}   ")

        return all_w, all_b, J_history

    def plot_decision_boundary(self, X, y, w, b, scaler, feature_names, label_encoder, use_polynomial=False, degree=6):
        """
        Plot the decision boundary for logistic regression.
        
        Parameters:
        X: array-like, feature matrix (original scale, 2 features)
        y: array-like, target vector
        w: array-like, learned weight parameters (without intercept)
        b: scalar, learned bias parameter
        scaler: StandardScaler, fitted scaler (only used if use_polynomial=False)
        feature_names: list, names of the two features
        label_encoder: LabelEncoder, fitted label encoder
        use_polynomial: bool, whether to use polynomial features (default False)
        degree: int, degree of polynomial features if use_polynomial=True (default 6)
        """
        # Plot data points
        plt.figure(figsize=(12, 8))
        for class_label in np.unique(y):
            mask = y == class_label
            plt.scatter(X[mask, 0], X[mask, 1], 
                    label=f'{label_encoder.inverse_transform([class_label])[0]}',
                    alpha=0.7, s=30, edgecolors='black', linewidth=0.5)
        
        # Create a mesh grid
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                            np.linspace(y_min, y_max, 100))
        
        if use_polynomial:
            # Polynomial features case - VECTORIZED
            # Flatten mesh grid for vectorized processing
            mesh_points = np.c_[xx.ravel(), yy.ravel()]
            
            # Vectorized polynomial feature mapping for all points at once
            # This is much faster than nested loops
            poly_features = map_feature(mesh_points[:, 0], mesh_points[:, 1], degree)
            
            # Vectorized sigmoid computation for all points
            z = self.sigmoid(poly_features @ w + b)
            
            # Reshape back to mesh grid shape
            z = z.reshape(xx.shape)
            
            # Plot probability map
            plt.contourf(xx, yy, z, levels=50, alpha=0.6, cmap='RdYlBu')
            plt.colorbar(label='Probability')
            
            # Plot decision boundary
            plt.contour(xx, yy, z, levels=[0.5], colors='black', linewidths=2, linestyles='--')
        else:
            # Linear case: z = w[0]*x1_scaled + w[1]*x2_scaled + b
            # We need to work in original feature space, then scale
            mesh_points = np.c_[xx.ravel(), yy.ravel()]
            mesh_points_scaled = scaler.transform(mesh_points)
            # Calculate z = w @ x_scaled + b
            Z = self.sigmoid(mesh_points_scaled @ w + b)
            Z = Z.reshape(xx.shape)
            
            # Plot probability map
            plt.contourf(xx, yy, Z, levels=50, alpha=0.6, cmap='RdYlBu')
            plt.colorbar(label='Probability')
            
            # Plot decision boundary
            plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2, linestyles='--')
        
        plt.xlabel(feature_names[0])
        plt.ylabel(feature_names[1])
        title = 'Decision Boundary for Logistic Regression'
        if use_polynomial:
            title += f' (Polynomial degree {degree})'
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def predict(self, X, w, b): 
        """
        Predict whether the label is 0 or 1 using learned logistic
        regression parameters w (VECTORIZED VERSION)
        
        Args:
        X : (ndarray Shape (m, n))
        w : (array_like Shape (n,))      Parameters of the model
        b : (scalar, float)              Parameter of the model

        Returns:
        p: (ndarray (m,))
            The predictions for X using a threshold at 0.5
        """
        # Vectorized computation: z = X @ w + b for all examples at once
        z = X @ w + b
        
        # Vectorized sigmoid computation
        f_wb = self.sigmoid(z)
        
        # Vectorized prediction: p = (f_wb >= 0.5) for all examples
        p = (f_wb >= 0.5).astype(int)
        
        return p

    def predict_multiclass_ova(self, X, all_w, all_b, num_classes):
        """
        Predicts the class label for multiclass OvA.
        """
        m = X.shape[0]
        
        # Store probabilities for each class (m examples, K classes)
        probabilities = np.zeros((m, num_classes))
        
        for i in range(num_classes):
            w_i = all_w[i]
            b_i = all_b[i]
            
            # Calculate z = X @ w + b
            z = X @ w_i + b_i
            
            # Calculate probability using your existing sigmoid function
            probabilities[:, i] = self.sigmoid(z)
            
        # The final prediction is the class with the highest probability
        # np.argmax returns the index (class label) of the max value along axis 1 (the classes)
        predictions = np.argmax(probabilities, axis=1)
        
        return predictions, probabilities

    def predict_multiclass_softmax(self, X, all_w, all_b):
        """
        Predicts the class label for multiclass using Softmax Regression.
        
        Args:
        X (ndarray (m, n)): Input data, m examples, n features
        all_w (ndarray (K, n)): Weights for K classes (K=7)
        all_b (ndarray (K,)): Biases for K classes
        
        Returns:
        predictions (ndarray (m,)): The index of the predicted class for each example.
        probabilities (ndarray (m, K)): Softmax probabilities for all classes.
        """
        
        # 1. Calculate the linear score (Z) for all classes at once.
        # The linear calculation is Z = X @ W + b
        
        # Reshape all_b to (1, K) for proper broadcasting with Z (m, K)
        b_reshaped = all_b.reshape(1, -1) 
        
        # Z (m, n) @ all_w.T (n, K) -> Z (m, K) matrix of logits (scores)
        Z = X @ all_w.T + b_reshaped
        
        # 2. Apply the Softmax function to convert scores to probabilities (A)
        probabilities = my_softmax(Z)
        
        # 3. The final prediction is the class with the highest probability index
        predictions = np.argmax(probabilities, axis=1)
        
        return predictions, probabilities
        
    def evaluate(self, X, y, w, b):
        y_pred = self.predict(X, w, b)
        accuracy = accuracy_score(y, y_pred)
        cost = self.compute_cost(X, y, w, b)
        return y_pred, accuracy, cost

    def confusion_matrix(self, y, y_pred, label_encoder):
        cm = confusion_matrix(y, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=[label_encoder.inverse_transform([0])[0], 
                                label_encoder.inverse_transform([1])[0]],
                    yticklabels=[label_encoder.inverse_transform([0])[0], 
                                label_encoder.inverse_transform([1])[0]])
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix - Logistic Regression')
        plt.tight_layout()
        plt.show()
        return cm
    
    def plot_cost_history(self, costs_history):
        plt.plot(costs_history)
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.title('Cost History')
        plt.show()
    
    def plot_weights_history(self, weights_history):
        plt.plot(weights_history)
        plt.xlabel('Iteration')
        plt.ylabel('Weights')