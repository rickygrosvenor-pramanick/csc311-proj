"""
Implements bagging ensemble model using models from Q1, Q2, and Q3.
"""

from utils import (
    load_train_csv,
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
)
import numpy as np
from scipy.sparse import coo_matrix
from sklearn.impute import KNNImputer
import torch
from torch.autograd import Variable

from item_response import *
from neural_network import train, AutoEncoder


def convert_dict_to_sparse(data: dict[str, list[int]]):
    question_id = np.array(data["question_id"])
    user_id = np.array(data["user_id"])
    is_correct = np.array(data["is_correct"])

    num_users = user_id.max() + 1
    num_questions = question_id.max() + 1
    sparse_matrix = coo_matrix(
        (is_correct, (user_id, question_id)), shape=(num_users, num_questions)
    )
    return sparse_matrix.tocsc()


def load_data_NN(sparse_matrix, base_path="./data"):
    """Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_matrix = sparse_matrix.toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data


def sample_with_replacement(data: dict[str, list[int]]) -> dict[str, list[int]]:
    num_data_points = len(data["is_correct"])
    sampled_data = {}
    for key in data:
        sampled_data[key] = []

    for _ in range(num_data_points):
        rand_idx = np.random.randint(0, num_data_points)
        for key in data:
            sampled_data[key].append(data[key][rand_idx])

    return sampled_data


def evaluate_ensemble(train_data, valid_data, theta, beta, model, knn_matrix) -> float:
    model.eval()
    correct, total = 0, 0

    for i, u in enumerate(valid_data["user_id"]):
        q = valid_data["question_id"][i]

        # IRT
        irt_prediction = sigmoid(theta[u] - beta[q])

        # NN
        inputs = Variable(train_data[u]).unsqueeze(0)
        nn_output = model(inputs)
        nn_prediction = nn_output[0][q].item()

        # KNN
        knn_prediction = knn_matrix[u, q]

        combined_prediction = (irt_prediction + nn_prediction + knn_prediction) / 3

        if (combined_prediction >= 0.5) == valid_data["is_correct"][i]:
            correct += 1
        total += 1

    return correct / float(total)


def main():
    train_data = load_train_csv()
    valid_data = load_valid_csv()
    test_data = load_public_test_csv()

    matrix_from_file = load_train_sparse()
    print(type(matrix_from_file))
    print(matrix_from_file.shape)

    # Hyperparameters (optimal ones found in Q1 - Q3)
    irt_lr = 0.003
    irt_n_iterations = 100

    nn_k = 50
    nn_lr = 0.01
    nn_num_epoch = 50
    nn_lamb = 0.001

    knn_k = 11

    # 1. Sample 3 diff datasets with replacement
    train1 = sample_with_replacement(train_data)
    train2 = sample_with_replacement(train_data)
    train3 = sample_with_replacement(train_data)

    matrix_dict = convert_dict_to_sparse(train1)
    print(type(matrix_dict))
    print(matrix_dict.shape)

    # 2. Train models

    # Model 1 - IRT
    irt_ret = irt(
        train1,
        valid_data,
        len(set(train1["user_id"])),
        len(set(train1["question_id"])),
        irt_lr,
        irt_n_iterations,
    )
    theta1, beta1 = irt_ret[0], irt_ret[1]

    # Model 2 - NN
    sparse_train2_matrix = convert_dict_to_sparse(train2)
    zero_train_matrix, train_matrix, valid_data, test_data = load_data_NN(
        sparse_train2_matrix
    )
    model = AutoEncoder(train_matrix.shape[1], nn_k)

    train(
        model, nn_lr, nn_lamb, train_matrix, zero_train_matrix, valid_data, nn_num_epoch
    )

    # Model 3 - KNN
    sparse_train3_matrix = convert_dict_to_sparse(train3)
    knn_imputer = KNNImputer(n_neighbors=knn_k)
    knn_matrix = knn_imputer.fit_transform(sparse_train3_matrix.toarray())

    # 3. Evaluate on test data
    test_acc = evaluate_ensemble(
        zero_train_matrix, test_data, theta1, beta1, model, knn_matrix
    )
    print(f"Final test accuracy: {test_acc}")


if __name__ == "__main__":
    main()
