from torch.utils.data import Dataset, DataLoader

# dataset loaded from pandas dataframe
class QuestionDataset(Dataset):
	def __init__(self, df):
		self.user_ids = df['user_id'].values
		self.question_ids = df['question_id'].values
		self.is_correct = df['is_correct'].values

	def __len__(self):
		return len(self.is_correct)

	def __getitem__(self, idx):
		return self.user_ids[idx], self.question_ids[idx], self.is_correct[idx]

# helper function to initialize datasets and train and val dataloaders
def initialize_dataloaders(train_df, val_df, batch_size):
	train_dataset = QuestionDataset(train_df)
	train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

	val_dataset = QuestionDataset(val_df)
	val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

	return train_dataloader, val_dataloader