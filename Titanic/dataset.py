from torch.utils import data

class Dataset(data.Dataset):
  # Characterizes a dataset for PyTorch
  def __init__(self, features, labels):
        # Initialization
        self.labels = labels
        self.features = features

  def __len__(self):
        # Denotes the total number of samples
        return len(self.features)

  def __getitem__(self, index):
        # Generates one sample of data
        # Select sample
        X = self.features[index]
        # Get label
        y = self.labels[index]

        return X, y