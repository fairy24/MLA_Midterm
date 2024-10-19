import numpy as np
from utils import *
import matplotlib.pyplot as plt


def sigmoid(x):
    """ Apply sigmoid function. """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta, reg_lambda=0):
    """ Compute the negative log-likelihood with L2 regularization. """
    log_likelihood = 0.
    for i in range(len(data['user_id'])):
        user = data['user_id'][i]
        question = data['question_id'][i]
        c_ij = data['is_correct'][i]
        sig = sigmoid(theta[user] - beta[question])
        log_likelihood += c_ij * np.log(sig) + (1 - c_ij) * np.log(1 - sig)
    reg_term = reg_lambda * (np.sum(theta ** 2) + np.sum(beta ** 2))
    return -(log_likelihood - reg_term)


def update_theta_beta(data, lr, theta, beta, batch_size=None):
    """ Update theta and beta using mini-batch gradient descent. """
    if batch_size:
        data_batches = create_mini_batches(data, batch_size)
    else:
        data_batches = [data]  # Single full batch

    for batch in data_batches:
        user_ids = np.array(batch["user_id"])
        question_ids = np.array(batch["question_id"])
        is_correct = np.array(batch["is_correct"])

        sigmoids = sigmoid(theta[user_ids] - beta[question_ids])
        part_theta = np.bincount(user_ids, weights=(is_correct - sigmoids), minlength=theta.shape[0])
        part_beta = np.bincount(question_ids, weights=(sigmoids - is_correct), minlength=beta.shape[0])

        theta += lr * part_theta
        beta += lr * part_beta

    return theta, beta


def create_mini_batches(data, batch_size):
    """ Create mini-batches from the data. """
    indices = np.random.permutation(len(data["is_correct"]))
    for start_idx in range(0, len(data["is_correct"]), batch_size):
        idx = indices[start_idx:start_idx + batch_size]
        yield {key: np.array(value)[idx] for key, value in data.items()}


def irt(data, val_data, lr, iterations, batch_size=None, reg_lambda=0, early_stopping_patience=None):
    """ Train IRT model with regularization, early stopping, and mini-batch gradient descent. """
    theta = np.full(542, 0.5)  # Initialize theta (user ability)
    beta = np.zeros(1774)  # Initialize beta (question difficulty)

    val_acc_lst = []
    train_log_like = []
    val_log_like = []

    best_val_lld = float('inf')
    no_improvement_counter = 0

    for i in range(iterations):
        # Calculate negative log-likelihood for training and validation sets
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta, reg_lambda=reg_lambda)
        val_neg_lld = neg_log_likelihood(val_data, theta=theta, beta=beta, reg_lambda=reg_lambda)
        score = evaluate(val_data, theta, beta)

        # Store the log-likelihood and accuracy
        train_log_like.append(neg_lld)
        val_log_like.append(val_neg_lld)
        val_acc_lst.append(score)

        # Update theta and beta using gradient descent
        theta, beta = update_theta_beta(data, lr, theta, beta, batch_size)

        # Print progress every 100 iteration
        if i % 100 == 0:
            print(f"Iteration {i}: Train NLL = {neg_lld}, Val NLL = {val_neg_lld}, Val Accuracy = {score}")

        # Early stopping
        if val_neg_lld < best_val_lld:
            best_val_lld = val_neg_lld
            no_improvement_counter = 0
        else:
            no_improvement_counter += 1

        if early_stopping_patience and no_improvement_counter >= early_stopping_patience:
            print(f"Early stopping at iteration {i}")
            break

    return theta, beta, val_acc_lst, train_log_like, val_log_like


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy. """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) / len(data["is_correct"])


def plot_log_likelihood(train_log_like, val_log_like):
    """ Plot the negative log-likelihood for training and validation sets. """
    plt.figure(figsize=(8, 6))
    plt.plot(range(len(train_log_like)), train_log_like, label="Train NLL", color='blue')
    plt.plot(range(len(val_log_like)), val_log_like, label="Validation NLL", color='orange')
    plt.xlabel("Iterations")
    plt.ylabel("Negative Log-Likelihood")
    plt.title("Negative Log-Likelihood Over Iterations")
    plt.legend()
    plt.show()


def plot_probability_of_response(theta, beta, question_ids):
    """ Plot the probability of response based on student ability (theta) and selected questions (beta). """
    theta_vals = np.linspace(-3, 3, 100)  # Simulated ability range

    plt.figure(figsize=(8, 6))
    for q_id in question_ids:
        beta_q = beta[q_id]
        prob = sigmoid(theta_vals - beta_q)  # Probability of a correct response
        plt.plot(theta_vals, prob, label=f"Question {q_id}")

    plt.xlabel("Student Ability (theta)")
    plt.ylabel("Probability of Correct Response")
    plt.title("Probability of Correct Response vs. Student Ability")
    plt.legend()
    plt.show()


def main():
    # Load the datasets
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    # Set hyperparameters
    num_iterations = 200
    lr = 0.00255
    batch_size = 128
    reg_lambda = 0.001
    early_stopping_patience = 10

    # Train the model using IRT with the enhancements
    theta, beta, val_acc_lst, train_log_like, val_log_like = irt(
        train_data, val_data, lr, num_iterations, batch_size, reg_lambda, early_stopping_patience
    )

    # Plot the negative log-likelihoods for both training and validation sets
    plot_log_likelihood(train_log_like, val_log_like)

    # Plot the probability of correct responses for selected questions
    plot_probability_of_response(theta, beta, question_ids=[1, 2, 3])  # Select a few questions to visualize

    # Evaluate on validation and test sets
    print(f"Validation accuracy: {evaluate(val_data, theta, beta)}")
    print(f"Test accuracy: {evaluate(test_data, theta, beta)}")


if __name__ == "__main__":
    main()