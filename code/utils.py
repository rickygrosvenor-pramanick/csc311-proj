import pandas as pd
import torch
import os

# gets n_students and n_questions from metadata files. These are required to initialize the embeddings
def get_counts(student_meta: pd.DataFrame, question_meta: pd.DataFrame) -> tuple:
	n_students = max(student_meta["user_id"]) + 1
	n_questions = max(question_meta["question_id"]) + 1
	return n_students, n_questions


# Loads the desired attributes (gender, premium_pupil) from the student metadata csv file
def load_student_meta_tensor(student_meta: pd.DataFrame) -> torch.Tensor:
	n_students = len(student_meta)
	student_meta_tensor = torch.zeros(n_students, 2)
	user_id = torch.tensor(student_meta['user_id'].values, dtype=torch.int32)
	gender = torch.tensor(student_meta['gender'].values, dtype=torch.float32)
	premium_pupil = torch.tensor(
		student_meta['premium_pupil'].fillna(-1.0).values, dtype=torch.float32
	)
	student_meta_tensor[user_id, 0] = gender
	student_meta_tensor[user_id, 1] = premium_pupil
	return student_meta_tensor


# loads all CSVs from dataset directory
def load_dfs(data_dir_path: str) -> tuple:
	train_df = pd.read_csv(os.path.join(data_dir_path, "train_data.csv"))
	val_df = pd.read_csv(os.path.join(data_dir_path, "valid_data.csv"))
	student_meta = pd.read_csv(os.path.join(data_dir_path, "student_meta.csv"))
	question_meta = pd.read_csv(os.path.join(data_dir_path, "question_meta.csv"))
	subject_meta = pd.read_csv(os.path.join(data_dir_path, "subject_meta.csv"))
	return train_df, val_df, student_meta, question_meta, subject_meta