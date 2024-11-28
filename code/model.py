import torch
from torch import nn, optim

# model class. predicts based on student embedding, question embedding and student metadata
class StudentQuestionNet(nn.Module):
	def __init__(self, student_embed_dim, question_embed_dim, student_meta_dim, hidden_layers, dropout_p=0.3):
		super(StudentQuestionNet, self).__init__()
		input_dim = student_embed_dim + question_embed_dim + student_meta_dim
		layers = []

		# add arbitrary number of layers based on input parameter hidden_layers. Layer as defined in report diagrams
		for hidden_dim in hidden_layers:
			layers.append(nn.Linear(input_dim, hidden_dim))
			layers.append(nn.BatchNorm1d(hidden_dim))
			layers.append(nn.ReLU())
			layers.append(nn.Dropout(dropout_p))
			input_dim = hidden_dim

		layers.append(nn.Linear(input_dim, 1))
		self.network = nn.Sequential(*layers)
	
	def forward(self, student_embed, question_embed, student_meta):
		combined = torch.cat([student_embed, question_embed, student_meta], dim=-1)
		return self.network(combined)


# initializes model, embeddings and optimizers in one helper function
def initialize_model_and_optimizers(n_students, n_questions, student_embed_dim, question_embed_dim, student_meta_dim,
									hidden_layers, dropout_p, learning_rate, device):
	student_embed = nn.Parameter(torch.randn(n_students, student_embed_dim).to(device))
	question_embed = nn.Parameter(torch.randn(n_questions, question_embed_dim).to(device))

	model = StudentQuestionNet(
		student_embed_dim=student_embed_dim,
		question_embed_dim=question_embed_dim,
		student_meta_dim=student_meta_dim,
		hidden_layers=hidden_layers,
		dropout_p=dropout_p
	).to(device)

	model_optimizer = optim.Adam(model.parameters(), lr=learning_rate)
	embed_optimizer = optim.Adam([student_embed, question_embed], lr=learning_rate)

	return model, student_embed, question_embed, model_optimizer, embed_optimizer