import argparse
import pandas as pd
import torch
import json
import os
import glob
import torch.nn.functional as F

from model import StudentQuestionNet
from dataset import QuestionDataset
from utils import load_student_meta_tensor

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--checkpoint_folder", type=str, required=True, help="Path to checkpoint folder containing .pt files and hyperparameters.json")
parser.add_argument("-e", "--eval_dataset", type=str, required=True, help="Path to evaluation dataset CSV file")
parser.add_argument("-s", "--student_meta", type=str, required=True, help="Path to student metadata csv file")


# loads model and embeddings given a checkpoint folder
def load_checkpoint(checkpoint_folder: str):
	hp_file = os.path.join(checkpoint_folder, "hyperparameters.json")
	student_embeddings_file = os.path.join(checkpoint_folder, "student_embed.pt")
	question_embeddings_file = os.path.join(checkpoint_folder, "question_embed.pt")
	model_file = glob.glob(f"{checkpoint_folder}/epoch*")[0]

	f = open(hp_file, "r")
	hp = json.load(f)

	student_embed = torch.load(student_embeddings_file, weights_only=True)
	question_embed = torch.load(question_embeddings_file, weights_only=True)
	model = StudentQuestionNet(
		student_embed_dim=hp["student_embed_dim"],
		question_embed_dim=hp["question_embed_dim"],
		student_meta_dim=2,
		hidden_layers=hp["hidden_layers"],
		dropout_p=hp["dropout_p"]
	)
	model.load_state_dict(torch.load(model_file, weights_only=True))
	model.eval()
	return model, student_embed, question_embed


def load_dataset(dataset_path: str):
	df = pd.read_csv(dataset_path)
	dataset = QuestionDataset(df)
	return dataset


if __name__ == "__main__":
	args = parser.parse_args()
	model, student_embed, question_embed = load_checkpoint(args.checkpoint_folder)
	dataset = load_dataset(args.eval_dataset)
	student_meta_tensor = load_student_meta_tensor(pd.read_csv(args.student_meta))

	correct = 0
	true_labels = torch.zeros(len(dataset))
	pred_probs = torch.zeros(len(dataset))
	with torch.no_grad():
		for i, (uid, qid, is_correct) in enumerate(dataset):
			print(f"Computing metrics ... {100 * (i / len(dataset)):.3f}%", end="\r")

			s = student_embed[uid].unsqueeze(0)
			q = question_embed[qid].unsqueeze(0)
			s_meta = student_meta_tensor[uid].unsqueeze(0)
			pred = F.sigmoid(model(s, q, s_meta)).item()
			pred_probs[i] = pred
			true_labels[i] = is_correct

	acc = ((pred_probs > 0.5) == true_labels).sum() / len(dataset)
	bce_loss = F.binary_cross_entropy(pred_probs, true_labels)
	print(f"Accuracy: {acc}")
	print(f"BCE Loss: {bce_loss}")

