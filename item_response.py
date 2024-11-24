from utils import (
	load_train_csv,
	load_valid_csv,
	load_public_test_csv,
	load_train_sparse,
)
import numpy as np
import matplotlib.pyplot as plt # imported for generating graphs


def sigmoid(x):
	"""Apply sigmoid function."""
	return np.exp(x) / (1 + np.exp(x))

def neg_log_likelihood(data, theta, beta):
	"""Compute the negative log-likelihood.

	You may optionally replace the function arguments to receive a matrix.

	:param data: A dictionary {user_id: list, question_id: list,
	is_correct: list}
	:param theta: Vector of shape (N_students,)
	:param beta: Vector of shape (N_questions,)
	:return: float
	"""
	uids, qids, correct = [np.array(data[k]) for k in ["user_id", "question_id", "is_correct"]]
	theta_minus_beta = theta[uids] - beta[qids]
	log_likelihood = np.sum(correct * theta_minus_beta - np.log(1 + np.exp(theta_minus_beta)))
	return -log_likelihood / len(uids) # returning mean neg_lld (otherwise the graphs of train vs val are not on the same scale)

def update_theta_beta(data, lr, theta, beta):
	"""Update theta and beta using gradient descent.

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
	uids, qids, correct = [np.array(data[k]) for k in ["user_id", "question_id", "is_correct"]]
	frac = 1 / (1 + np.exp(beta[qids] - theta[uids])) # common fractional part in beta_grad and theta_grad
	
	# filling beta and theta grad iteratively
	theta_grad = np.zeros_like(theta)
	beta_grad = np.zeros_like(beta)
	for i, (u, q, c) in enumerate(zip(uids, qids, correct)):
		theta_grad[u] += frac[i] - c
		beta_grad[q] -= frac[i] - c
	
	# parameter update rules (regular grad desc)
	new_theta = theta - lr * theta_grad
	new_beta = beta - lr * beta_grad
	return new_theta, new_beta


def irt(data, val_data, n_students, n_questions, lr, iterations):
	"""Train IRT model.

	You may optionally replace the function arguments to receive a matrix.

	:param data: A dictionary {user_id: list, question_id: list,
	is_correct: list}
	:param val_data: A dictionary {user_id: list, question_id: list,
	is_correct: list}
	:param lr: float
	:param iterations: int
	:return: (theta, beta, val_acc_lst)
	"""
	theta = np.random.randn(n_students)
	beta = np.random.randn(n_questions)
	print(theta.shape, beta.shape)

	val_acc_lst = []
	train_lld_list = []
	val_lld_list = []

	for i in range(iterations):
		score = evaluate(data=val_data, theta=theta, beta=beta)
		val_acc_lst.append(score)

		# compute and save train and val neg_llds
		neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
		val_neg_lld = neg_log_likelihood(val_data, theta=theta, beta=beta)
		train_lld_list.append(neg_lld)
		val_lld_list.append(val_neg_lld)
		print("Iteration: [{}/{}] \t NLLK: {} \t Score: {}".format(i, iterations, neg_lld, score))

		# update parameters with computed gradients
		theta, beta = update_theta_beta(data, lr, theta, beta)

	return theta, beta, val_acc_lst, train_lld_list, val_lld_list


def evaluate(data, theta, beta):
	"""Evaluate the model given data and return the accuracy.
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
	return np.sum((data["is_correct"] == np.array(pred))) / len(data["is_correct"])


def main():
	train_data = load_train_csv("./data")
	# You may optionally use the sparse matrix.
	# sparse_matrix = load_train_sparse("./data")
	val_data = load_valid_csv("./data")
	test_data = load_public_test_csv("./data")

	n_students = max(max(train_data["user_id"]), max(val_data["user_id"]), max(test_data["user_id"])) + 1
	n_questions = max(max(train_data["question_id"]), max(val_data["question_id"]), max(test_data["question_id"])) + 1
	print(f"N_students: {n_students}, N_questions: {n_questions}")

	np.random.seed(311)
	lr = 0.003
	n_iterations = 100
	theta, beta, val_acc, train_lld, val_lld = irt(train_data, val_data, n_students, n_questions, lr, n_iterations)

	# plotting mean negative log likelihood for train and val
	plt.plot(train_lld, c="r", label="train")
	plt.plot(val_lld, c="b", label="validation")
	plt.xlabel("iteration")
	plt.ylabel("mean negative log-likelihood")
	plt.legend()
	plt.savefig("q2b-fig.jpg")
	plt.clf()
	plt.close()

	final_val_acc = evaluate(val_data, theta, beta)
	final_test_acc = evaluate(test_data, theta, beta)
	print(f"final validation accuracy: {final_val_acc}, final test accuracy: {final_test_acc}")

	# part d
	xvals = np.linspace(-5, 5, 100) # range of theta values on x axis
	j3 = np.random.randint(0, len(beta), 3) # 3 random questions
	plt.figure()
	for j in j3:
		exp_theta_minus_beta = np.exp(xvals - beta[j])
		p = exp_theta_minus_beta / (1 + exp_theta_minus_beta)
		plt.plot(xvals, p, label=f'Question {j}')

	plt.xlabel(r'$\theta$ (Student Ability)')
	plt.ylabel('$p(c_{ij} = 1)$')
	plt.legend()
	plt.savefig("q2d-fig.jpg")


if __name__ == "__main__":
	main()
