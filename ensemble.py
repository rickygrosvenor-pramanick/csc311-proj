# TODO: complete this file.
"""
Implements bagging ensemble model using models from Q1, Q2, and Q3.
"""

from utils import (
    load_train_csv,
    load_valid_csv,
    load_public_test_csv,
)
import numpy as np
from item_response import *


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


def ensemble(train_data: dict[str, list[int]], valid_data: dict[str, list[int]]):
    # sample with replacement 3 times
    train_data1 = sample_with_replacement(train_data)
    train_data2 = sample_with_replacement(train_data)
    train_data3 = sample_with_replacement(train_data)

    # hyperparams
    np.random.seed(311)
    lr = 0.003
    n_iterations = 100

    theta1, beta1, val_acc, train_lld, val_lld = irt(
        train_data1,
        valid_data,
        len(set(train_data1["user_id"])),
        len(set(train_data1["question_id"])),
        lr,
        n_iterations,
    )

    theta2, beta2, val_acc, train_lld, val_lld = irt(
        train_data2,
        valid_data,
        len(set(train_data2["user_id"])),
        len(set(train_data2["question_id"])),
        lr,
        n_iterations,
    )

    theta3, beta3, val_acc, train_lld, val_lld = irt(
        train_data3,
        valid_data,
        len(set(train_data3["user_id"])),
        len(set(train_data3["question_id"])),
        lr,
        n_iterations,
    )

    return [(theta1, beta1), (theta2, beta2), (theta3, beta3)]


def evaluate(data: dict[str, list[int]], theta_betas: list) -> float:
    pred = []
    for i, q in enumerate(data["question_id"]):
        curr_pred = []
        for theta, beta in theta_betas:
            u = data["user_id"][i]
            x = (theta[u] - beta[q]).sum()
            p_a = sigmoid(x)
            curr_pred.append(p_a >= 0.5)
        pred.append((sum(curr_pred) / len(curr_pred)) >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) / len(data["is_correct"])


def main():
    train_data = load_train_csv()
    valid_data = load_valid_csv()
    test_data = load_public_test_csv()

    theta_betas = ensemble(train_data, valid_data)
    test_acc = evaluate(test_data, theta_betas)
    print(f"Final test accuracy: {test_acc}")


if __name__ == "__main__":
    main()
