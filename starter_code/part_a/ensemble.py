from scipy.stats import mode
from sklearn.impute import KNNImputer
from sklearn.utils import resample
from starter_code.utils import load_train_sparse, load_valid_csv, load_public_test_csv
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from neural_network import AutoEncoder, load_data, train
import os
import sys
from item_response import irt, sigmoid

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from item_response import irt
from utils import *

# Neural Network model definition
class AutoEncoder(nn.Module):
    def __init__(self, num_question, k=100):
        super(AutoEncoder, self).__init__()
        self.g = nn.Linear(num_question, k)  # Use torch.nn.Linear
        self.h = nn.Linear(k, num_question)  # Use torch.nn.Linear

    def get_weight_norm(self):
        g_w_norm = torch.norm(self.g.weight, 2)
        h_w_norm = torch.norm(self.h.weight, 2)
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        out = torch.sigmoid(self.g(inputs))
        out = torch.sigmoid(self.h(out))
        return out


def train_neural_network(model, train_data, zero_train_data, valid_data, lr=0.1, lamb=0.01, num_epoch=20):
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]

    for epoch in range(num_epoch):
        train_loss = 0.
        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            nan_mask = nan_mask.reshape(-1)
            target[0][nan_mask] = output[0][nan_mask]

            loss = torch.sum((output - target) ** 2.) + lamb * 0.5 * model.get_weight_norm()
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

    return model


def neural_network_pred(model, data, zero_train_data):
    model.eval()
    pred = []
    for i, u in enumerate(data["user_id"]):
        inputs = Variable(zero_train_data[u]).unsqueeze(0)
        output = model(inputs)
        pred.append(output[0][data["question_id"][i]].item() >= 0.5)
    return pred


def resample(data):
    n_samples = len(data["user_id"])
    indices = np.random.choice(range(n_samples), size=n_samples, replace=True)
    resampled_data = {
        "user_id": np.array(data["user_id"])[indices],
        "question_id": np.array(data["question_id"])[indices],
        "is_correct": np.array(data["is_correct"])[indices]
    }
    return resampled_data


def irt_pred(data, theta, beta):
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return pred


def ensemble(train_matrix, train_data, val_data, test_data, hp):
    val_pred = []
    test_pred = []

    # KNN (user-based)
    knn_train_data = resample(train_data)
    knn_sparse_matrix = np.empty(train_matrix.shape)
    knn_sparse_matrix[:] = np.nan
    for i in range(len(knn_train_data["is_correct"])):
        knn_sparse_matrix[knn_train_data["user_id"][i], knn_train_data["question_id"][i]] \
            = knn_train_data["is_correct"][i]
    nbrs = KNNImputer(n_neighbors=hp["knn_user_k"])
    knn_mat = nbrs.fit_transform(knn_sparse_matrix)
    knn_acc = sparse_matrix_evaluate(val_data, knn_mat)
    print(f"KNN: Validation Accuracy: {knn_acc}")
    val_pred.append(sparse_matrix_predictions(val_data, knn_mat))
    test_pred.append(sparse_matrix_predictions(test_data, knn_mat))

    # IRT
    irt_train_data = resample(train_data)
    theta, beta, meta, extra, _ = irt(irt_train_data, val_data, hp["irt_lr"], hp["irt_iter"])
    irt_acc = evaluate(val_data, irt_pred(val_data, theta, beta))
    print(f"IRT: Validation Accuracy: {irt_acc}")
    val_pred.append(irt_pred(val_data, theta, beta))
    test_pred.append(irt_pred(test_data, theta, beta))

    # Neural Network
    zero_train_matrix = torch.FloatTensor(train_matrix.copy())
    zero_train_matrix[np.isnan(train_matrix)] = 0
    nn_model = AutoEncoder(train_matrix.shape[1], hp["nn_k"])
    nn_model = train_neural_network(nn_model, torch.FloatTensor(train_matrix), zero_train_matrix, val_data,
                                    lr=hp["nn_lr"], lamb=hp["nn_lamb"], num_epoch=hp["nn_epoch"])
    nn_acc = evaluate(val_data, neural_network_pred(nn_model, val_data, zero_train_matrix))
    print(f"Neural Network: Validation Accuracy: {nn_acc}")
    val_pred.append(neural_network_pred(nn_model, val_data, zero_train_matrix))
    test_pred.append(neural_network_pred(nn_model, test_data, zero_train_matrix))

    return val_pred, test_pred


def main():
    # Load the training, validation, and test datasets
    train_data = load_train_csv(r"..\data")
    sparse_matrix = load_train_sparse(r"..\data").toarray()
    val_data = load_valid_csv(r"..\data")
    test_data = load_public_test_csv(r"..\data")


    # hyper-parameters
    hp = {
        # KNN
        "knn_user_k": 11,
        # IRT
        "irt_lr": 0.006,
        "irt_iter": 100,
        # Neural Network
        "nn_k": 10,
        "nn_lr": 0.1,
        "nn_lamb": 0.01,
        "nn_epoch": 20
    }

    # ensemble
    val_pred, test_pred = ensemble(sparse_matrix, train_data, val_data, test_data, hp)
    # the mean of predictions from the 3 models
    mean_val_pred = np.mean(np.array(val_pred), axis=0)
    mean_test_pred = np.mean(np.array(test_pred), axis=0)
    # accuracy of combined prediction
    val_acc = evaluate(val_data, mean_val_pred)
    test_acc = evaluate(test_data, mean_test_pred)
    print("Final Ensembled Results:")
    print(f"Validation Accuracy: {val_acc}")
    print(f"Test Accuracy: {test_acc}")


if __name__ == "__main__":
    main()
