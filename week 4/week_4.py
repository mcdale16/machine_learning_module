import numpy as np
import matplotlib.pyplot as plt

def linear_classify(x, theta, theta_0):
    """Uses the given theta, theta_0, to linearly classify the given data x. This is our hypothesis or hypothesis class.

    :param x:
    :param theta:
    :param theta_0:
    :return: 1 if the given x is classified as positive, -1 if it is negative, and 0 if it lies on the hyperplane.
    """
    calc = np.dot(theta.T, x) + theta_0
    return np.sign(calc)
    pass


def Loss(prediction, actual):
    """Computes the loss between the given prediction and actual values.

    :param prediction:
    :param actual:
    :return:
    """
    return 0 if prediction * actual > 0 else 1
    pass


def E_n(h, data, labels, L, theta, theta_0):
    """Computes the error for the given data using the given hypothesis and loss.

    :param h: Hypothesis class, for example a linear classifier.
    :param data: A d x n matrix where d is the number of data dimensions and n the number of examples.
    :param labels: A 1 x n matrix with the label (actual value) for each data point.
    :param L: A loss function to compute the error between the prediction and label.
    :param theta:
    :param theta_0:
    :return:
    """
    (d, n) = data.shape
    total_loss = 0 
    for i in range(n):
        x_i = data[:, i: i + 1]
        y_i = labels[0, i]
        prediction = h(x_i, theta, theta_0)
        total_loss += L(prediction, y_i)
    return total_loss / n
    pass


def random_linear_classifier(data, labels, params={}, hook=None):
    """

    :param data: A d x n matrix where d is the number of data dimensions and n the number of examples.
    :param labels: A 1 x n matrix with the label (actual value) for each data point.
    :param params: A dict, containing a key T, which is a positive integer number of steps to run
    :param hook: An optional hook function that is called in each iteration of the algorithm.
    :return:
    """

    k = params.get('k', 100)
    (d, n) = data.shape
    
    best_theta = np.zeros((d, 1))
    best_theta_0 = 0.0
    min_error = float('inf')

    for j in range(k):
        theta = np.random.uniform(-1, 1, (d, 1)) # random parameter vector
        theta_0 = np.random.uniform(-1, 1)

        error = E_n(linear_classify, data, labels, Loss, theta, theta_0) # calculate error for random parameter
        
        if error < min_error:
            min_error = error
            best_theta, best_theta_0 = theta, theta_0
            
        if hook: hook((best_theta, best_theta_0))
            
    return best_theta, best_theta_0
    pass


def perceptron(data, labels, params={}, hook=None):
    """The Perceptron learning algorithm.

    :param data: A d x n matrix where d is the number of data dimensions and n the number of examples.
    :param labels: A 1 x n matrix with the label (actual value) for each data point.
    :param params: A dict, containing a key T, which is a positive integer number of steps to run
    :param hook: An optional hook function that is called in each iteration of the algorithm.
    :return:
    """
    T = params.get('T', 100)
    (d, n) = data.shape

    theta = np.zeros((d, 1))
    theta_0 = 0.0 # initialise to 0s

    for t in range(T):
        changed = False
        for i in range(n):
            x_i = data[:, i:i+1]
            y_i = labels[0, i]

            # a point is misclassified if y_i * (theta.T @ x_i + theta_0) <= 0
            # points on hyperplane are also considered misclassified
            if y_i * (np.dot(theta.T, x_i) + theta_0) <= 0:
                theta = theta + y_i * x_i
                theta_0 = theta_0 + y_i
                changed = True
        
        if hook: hook((theta, theta_0))
            
        if not changed:
            break
    return theta, theta_0
    pass


def margin(data, labels, theta, theta_0):
    """Computes the geometric margin of the classifier on the dataset.
    """
    theta_norm = np.linalg.norm(theta)
    if theta_norm == 0:
        return 0
    # Calculate signed distances: y * (theta^T x + theta_0) / ||theta||
    distances = (labels * (np.dot(theta.T, data) + theta_0)) / theta_norm
    return np.min(distances)


def plot_separator(plot_axes, theta, theta_0, label=None, color=None):
    """Plots the linear separator defined by theta, theta_0, into the given plot_axes.

    :param plot_axes: Matplotlib Axes object
    :param theta:
    :param theta_0:
    """
    if theta[1, 0] == 0:
        return

    # Extract scalar values safely
    th0 = float(theta[0, 0])
    th1 = float(theta[1, 0])
    th0_val = float(theta_0)

    y_intercept = -th0_val / th1
    slope = -th0 / th1
    
    xmin, xmax = -20, 20
    p1_y = slope * xmin + y_intercept
    p2_y = slope * xmax + y_intercept

    # Plot the separator
    plot_axes.plot([xmin, xmax], [p1_y, p2_y], '-', label=label, color=color)
    
    # Plot the normal vector arrow
    mid_x, mid_y = (xmin + xmax) / 2, (p1_y + p2_y) / 2
    plot_axes.arrow(mid_x, mid_y, th0, th1, head_width=0.5, head_length=0.5)


if __name__ == '__main__':
    """
    We'll define data X with its labels y, plot the data, and then run either the random_linear_classifier or the
    perceptron learning algorithm, to find a hypothesis h from the class of linear classifiers.
    We then plot the best hypothesis, as well as compute the training error. 
    """

    # Generate synthetic data
    X = np.random.uniform(low=-5, high=5, size=(2, 20)) 
    y = np.sign(np.dot(np.transpose([[3], [4]]), X) + 6) 

    # Plot setup
    colors = np.choose(y > 0, np.transpose(np.array(['r', 'g']))).flatten()
    plt.ion()
    fig, ax = plt.subplots()
    ax.scatter(X[0, :], X[1, :], c=colors, marker='o', edgecolors='k', zorder=5)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.grid(True, which='both')
    ax.axhline(color='black', linewidth=0.5)
    ax.axvline(color='black', linewidth=0.5)
    ax.set_title("Linear Classification & Margin Analysis")

    # Run and plot RLC
    theta_rlc, theta_0_rlc = random_linear_classifier(X, y, {"k": 500})
    plot_separator(ax, theta_rlc, theta_0_rlc, label="RLC", color='blue')

    # Run and plot Perceptron
    theta_p, theta_0_p = perceptron(X, y, {"T": 100})
    plot_separator(ax, theta_p, theta_0_p, label="Perceptron", color='purple')
    
    ax.legend()

    # Margin printouts
    print(f"RLC Margin: {margin(X, y, theta_rlc, theta_0_rlc):.4f}")
    print(f"Perceptron Margin: {margin(X, y, theta_p, theta_0_p):.4f}")

    # Plot E_n over various k
    ks = [1, 10, 50, 100, 500, 1000]
    errors = []
    for k in ks:
        th, th0 = random_linear_classifier(X, y, {"k": k})
        errors.append(E_n(linear_classify, X, y, Loss, th, th0))

    plt.figure()
    plt.plot(ks, errors, 'o-')
    plt.xscale('log')
    plt.xlabel('k (Number of random guesses)')
    plt.ylabel('Training Error (E_n)')
    plt.title('Error vs. K for Random Linear Classifier')
    plt.grid(True)

    plt.ioff()
    plt.show()
    print("Finished.")