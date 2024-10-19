from starter_code.utils import load_train_sparse, load_valid_csv, load_public_test_csv

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = 0.
    assert len(data['user_id']) == len(data['question_id']) == len(data['is_correct'])
    for i in range(len(data['user_id'])):
        user = data['user_id'][i]
        question = data['question_id'][i]
        c_ij = data['is_correct'][i]
        sig = sigmoid(theta[user] - beta[question])
        log_lklihood +=  c_ij * np.log(sig) + (1 -  c_ij) * np.log(1 - sig)
    # Return negative log-likelihood
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """

    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    assert len(data['user_id']) == len(data['question_id']) == len(data['is_correct'])
    part_theta = np.zeros(theta.shape[0])
    part_beta = np.zeros(beta.shape[0])

    for i in range(len(data['is_correct'])):
        user = data['user_id'][i]
        question = data['question_id'][i]
        c_ij = data['is_correct'][i]
        sig = sigmoid(theta[user] - beta[question])
        part_theta[user] += c_ij - sig
        part_beta[question] += -c_ij + sig

    theta += lr * part_theta
    beta += lr * part_beta


    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    return theta, beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    theta = np.full(542,0.5)
    beta = np.zeros(1774)

    val_acc_lst = []
    train_log_like = []
    val_log_like = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        val_neg_lld = neg_log_likelihood(val_data, theta=theta, beta=beta)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        train_log_like.append(neg_lld)
        val_log_like.append(val_neg_lld)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst, train_log_like,val_log_like


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")

    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    num_iterations=200
    lr=0.00255
    theta, beta, val_acc_lst, train_log_like, val_log_like = irt(train_data, val_data, lr, num_iterations)

    print(f"hyperparameters: \n num_iterations = "
          f"{num_iterations}, learning rate = {lr}")
    print("val accuracy: ")
    print(evaluate(val_data, theta, beta))
    print("test accuracy: ")
    print(evaluate(test_data, theta, beta))

    # plot for train set neg log likelihood
    plt.plot([x for x in range(num_iterations)], train_log_like)
    plt.xlabel("number of iterations")
    plt.ylabel("ned log likelihood")
    plt.title("Negative Log likelihood for Train Set")
    plt.show()

    # plot for validation set neg log likelihood
    plt.plot([x for x in range(num_iterations)], val_log_like)
    plt.xlabel("number of iterations")
    plt.ylabel("neg log likelihood")
    plt.title("Negative Log likelihood for Validation Set")
    plt.show()

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################
    plt.scatter(theta, [sigmoid(t - beta[2]) for t in theta], label="q1")
    plt.scatter(theta, [sigmoid(t - beta[3]) for t in theta], label="q2")
    plt.scatter(theta, [sigmoid(t - beta[4]) for t in theta], label="q3")
    plt.xlabel("theta")
    plt.ylabel("probability p(c_ij = 1)")
    plt.title("Probability of Correct Response vs Theta")
    plt.legend()
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
