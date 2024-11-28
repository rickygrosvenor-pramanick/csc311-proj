import os, json, argparse, torch, sys
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import torch.nn.functional as F
from typing import Tuple
import pandas as pd


from utils import get_counts, load_student_meta_tensor, load_dfs
from model import initialize_model_and_optimizers
from dataset import initialize_dataloaders

parser = argparse.ArgumentParser()
# cli args for dataset, checkpoint and logging folders
parser.add_argument("--dataset_path", type=str, help="path to dataset folder containing csv files")
parser.add_argument("--checkpoint_path", type=str, default="checkpoints", help="path to checkpoints folder for saving models")
parser.add_argument("--logs_path", type=str, default="logs", help="path to logs folder for tensorboard tracking")

# cli args for hyperparams (default parameter args produce the results in report)
parser.add_argument("--epochs", type=int, default=200, help="training epochs")
parser.add_argument("--student_embed_dim", type=int, default=8, help="Dimension of student embeddings")
parser.add_argument("--question_embed_dim", type=int, default=16, help="Dimension of question embeddings")
parser.add_argument("--hidden_layers", type=int, nargs='+', default=[64, 16], help="List of hidden layer sizes")
parser.add_argument("--dropout_p", type=float, default=0.3, help="Dropout probability")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for optimization")
parser.add_argument("--device", type=str, default="cpu", help="Device for computation (e.g., 'cpu', 'cuda' or 'mps')")

# performs a single training epoch, reports metrics
def train_epoch(
    model: nn.Module,
    train_dataloader: DataLoader,
    student_embed: torch.Tensor,
    question_embed: torch.Tensor,
    student_meta_tensor: torch.Tensor,
    crit: nn.Module,
    model_optimizer: Optimizer,
    embed_optimizer: Optimizer,
    device: torch.device
) -> Tuple[float, float]:
	model.train()
	epoch_loss = 0.0
	correct = 0
	n = 0

	for uid_batch, qid_batch, target_batch in train_dataloader:
		model_optimizer.zero_grad()
		embed_optimizer.zero_grad()

		target_batch = target_batch.float().unsqueeze(1).to(device)

		user_embeds = student_embed[uid_batch.to(device)]
		question_embeds = question_embed[qid_batch.to(device)]
		student_meta = student_meta_tensor.to(device)[uid_batch]

		logits = model(user_embeds, question_embeds, student_meta)
		loss = crit(logits, target_batch)

		loss.backward()
		model_optimizer.step()
		embed_optimizer.step()

		epoch_loss += loss.item()
		preds = F.sigmoid(logits) > 0.5
		correct += (preds == target_batch).sum().item()
		n += target_batch.shape[0]

	return epoch_loss / len(train_dataloader), correct / n


# performs a single val epoch, reports metrics
def val_epoch(
    model: nn.Module,
    val_dataloader: DataLoader,
    student_embed: torch.Tensor,
    question_embed: torch.Tensor,
    student_meta_tensor: torch.Tensor,
    crit: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
	model.eval()
	epoch_loss = 0.0
	correct = 0
	n = 0

	with torch.no_grad():
		for uid_batch, qid_batch, target_batch in val_dataloader:
			target_batch = target_batch.float().unsqueeze(1).to(device)

			user_embeds = student_embed[uid_batch.to(device)]
			question_embeds = question_embed[qid_batch.to(device)]
			student_meta = student_meta_tensor.to(device)[uid_batch]

			logits = model(user_embeds, question_embeds, student_meta)
			loss = crit(logits, target_batch)

			epoch_loss += loss.item()
			preds = F.sigmoid(logits) > 0.5
			correct += (preds == target_batch).sum().item()
			n += target_batch.shape[0]

	return epoch_loss / len(val_dataloader), correct / n


# checkpoints model and embeddings if val_acc is better than previous best val_acc
def checkpoint(
    epoch: int,
    experiment_path: str,
    avg_val_loss: float,
    val_acc: float,
    model: nn.Module,
    student_embed: torch.Tensor,
    question_embed: torch.Tensor,
    best_val_acc: float
) -> float:
	if val_acc > best_val_acc:
		best_val_acc = val_acc

		for file in os.listdir(experiment_path):
			if file.endswith('.pt'):
				os.remove(os.path.join(experiment_path, file))

		model_filename = f'epoch{epoch}_val_loss{avg_val_loss:.5f}_val_acc{val_acc:.5f}.pt'
		model_save_path = os.path.join(experiment_path, model_filename)
		torch.save(model.state_dict(), model_save_path)

		torch.save(student_embed, os.path.join(experiment_path, 'student_embed.pt'))
		torch.save(question_embed, os.path.join(experiment_path, 'question_embed.pt'))

	return best_val_acc


# full training function
def train(
    checkpoint_dir: str,
    log_dir: str,
	epochs: int,
    n_students: int,
    n_questions: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    student_meta_tensor: torch.Tensor,
    student_embed_dim: int,
    question_embed_dim: int,
    hidden_layers: list[int],
    dropout_p: float,
    batch_size: int,
    learning_rate: float,
    device: str
) -> None:
	# finding a unique experiment name based on completed experiments
	experiment_num = 1
	while os.path.exists(os.path.join(checkpoint_dir, f'experiment{experiment_num}')):
		experiment_num += 1
	experiment_path = os.path.join(checkpoint_dir, f'experiment{experiment_num}')
	os.makedirs(experiment_path, exist_ok=True)

	# saving hyperparameters.json file so that we can rebuild the same model later and load weights
	hyperparameters = {
		'n_students': n_students,
		'n_questions': n_questions,
		'student_embed_dim': student_embed_dim,
		'question_embed_dim': question_embed_dim,
		'hidden_layers': hidden_layers,
		'dropout_p': dropout_p,
		'batch_size': batch_size,
		'learning_rate': learning_rate,
		'device': device
	}
	with open(os.path.join(experiment_path, 'hyperparameters.json'), 'w') as f:
		json.dump(hyperparameters, f, indent=4)

	# initializing dataloaders, models, optimizers and loss
	crit = torch.nn.BCEWithLogitsLoss()
	train_dataloader, val_dataloader = initialize_dataloaders(train_df, val_df, batch_size)
	model, student_embed, question_embed, model_optimizer, embed_optimizer = initialize_model_and_optimizers(
		n_students, n_questions, student_embed_dim, question_embed_dim, student_meta_tensor.shape[1],
		hidden_layers, dropout_p, learning_rate, device
	)

	# creating tensorboard writer for logging metrics and logging a model graph
	writer = SummaryWriter(os.path.join(log_dir, f'experiment{experiment_num}'))
	dummy_student_embed = nn.Parameter(torch.randn(32, student_embed_dim)).to(device)
	dummy_question_embed = nn.Parameter(torch.randn(32, question_embed_dim)).to(device)
	dummy_student_meta = nn.Parameter(torch.randn(32, student_meta_tensor.shape[1])).to(device)
	writer.add_graph(model, (dummy_student_embed, dummy_question_embed, dummy_student_meta))

	best_val_acc = -1
	for epoch in range(1, epochs + 1):
		avg_train_loss, train_acc = train_epoch(
			model, train_dataloader, student_embed, question_embed, student_meta_tensor,
			crit, model_optimizer, embed_optimizer, device
		)

		avg_val_loss, val_acc = val_epoch(
			model, val_dataloader, student_embed, question_embed, student_meta_tensor,
			crit, device
		)

		writer.add_scalar('Train/Loss', avg_train_loss, epoch)
		writer.add_scalar('Train/Accuracy', train_acc, epoch)
		writer.add_scalar('Validation/Loss', avg_val_loss, epoch)
		writer.add_scalar('Validation/Accuracy', val_acc, epoch)

		print(f"Epoch: {epoch} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {avg_val_loss:.5f} | Val Acc: {val_acc:.5f}")

		best_val_acc = checkpoint(
			epoch, experiment_path, avg_val_loss, val_acc, model, student_embed, question_embed, best_val_acc
		)

	writer.close()


if __name__ == "__main__":
	args = parser.parse_args()
	if not args.dataset_path:
		print("Error: --dataset_path is required.")
		sys.exit(1)

	train_df, val_df, student_meta, question_meta, subject_meta = load_dfs(args.dataset_path)
	n_students, n_questions = get_counts(student_meta, question_meta)
	student_meta_tensor = load_student_meta_tensor(student_meta)
	train(
		# datasets
		train_df = train_df,
		val_df = val_df,
		student_meta_tensor = student_meta_tensor,

		# hyperparamters
		epochs = args.epochs,
		n_students = n_students,
		n_questions = n_questions,
		student_embed_dim = args.student_embed_dim,
		question_embed_dim = args.question_embed_dim,
		hidden_layers = args.hidden_layers,
		dropout_p = args.dropout_p,
		batch_size = args.batch_size,
		learning_rate = args.learning_rate,
		device = args.device,

		# log and checkpoint dirs
		checkpoint_dir = args.checkpoint_path,
		log_dir = args.logs_path
	)
