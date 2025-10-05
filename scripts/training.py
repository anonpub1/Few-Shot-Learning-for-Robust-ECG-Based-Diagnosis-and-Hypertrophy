import torch
import torch.optim as optim
from model import TCNEncoder
from data_preprocessing import process_folder
from sklearn.utils.class_weight import compute_class_weight

device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_segments, train_labels = process_folder('path_to_data')

model = TCNEncoder().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(1, 51):
    model.train()
    pass
