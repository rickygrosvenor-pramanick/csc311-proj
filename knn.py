import numpy as np
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
from utils import (
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
    sparse_matrix_evaluate,
)


def knn_impute_by_user(matrix, valid_data, k):
    """Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy (user), k={}: {}".format(k, acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # Implement the function as described in the docstring.             #
    #####################################################################
    transposed_matrix = matrix.T
    nbrs = KNNImputer(n_neighbors=k)

    imputed_matrix = nbrs.fit_transform(transposed_matrix)
    restored_matrix = imputed_matrix.T

    acc = sparse_matrix_evaluate(valid_data, restored_matrix)
    print("Validation Accuracy (item), k={}: {}".format(k, acc))

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def main():
    sparse_matrix = load_train_sparse("./data").toarray()
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    #####################################################################
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################
    k_vals = [1, 6, 11, 16, 21, 26]
    max_k_user_acc, max_k_item_acc = -float("inf"), -float("inf")
    max_k_user, max_k_item = -1, -1
    user_accuracies = []
    item_accuracies = []

    for k in k_vals:
        curr_k_user_acc = knn_impute_by_user(sparse_matrix, val_data, k)
        curr_k_item_acc = knn_impute_by_item(sparse_matrix, val_data, k)
        user_accuracies.append(curr_k_user_acc)
        item_accuracies.append(curr_k_item_acc)

        if curr_k_user_acc > max_k_user_acc:
            max_k_user_acc = curr_k_user_acc
            max_k_user = k

        if curr_k_item_acc > max_k_item_acc:
            max_k_item_acc = curr_k_item_acc
            max_k_item = k

    user_test_acc_k_star = knn_impute_by_user(sparse_matrix, test_data, max_k_user)
    item_test_acc_k_star = knn_impute_by_item(sparse_matrix, test_data, max_k_item)
    print(f"User KNN test accuracy for k*={max_k_user}: {user_test_acc_k_star}")
    print(f"Item KNN test accuracy for k*={max_k_item}: {item_test_acc_k_star}")

    plt.figure(figsize=(10, 6))
    plt.plot(k_vals, user_accuracies, label="User-based KNN Accuracy", marker="o")
    plt.plot(k_vals, item_accuracies, label="Item-based KNN Accuracy", marker="s")

    plt.scatter(
        [max_k_user],
        [max_k_user_acc],
        color="blue",
        label=f"Max User Accuracy (k*={max_k_user})",
        zorder=5,
    )
    plt.scatter(
        [max_k_item],
        [max_k_item_acc],
        color="red",
        label=f"Max Item Accuracy (k*={max_k_item})",
        zorder=5,
    )

    plt.title("KNN Accuracy vs. Number of Neighbors (k)")
    plt.xlabel("Number of Neighbors (k)")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
