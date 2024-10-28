import os
import torch
import torch.nn as nn
import torch.optim as optim
import PyPDF2
import re
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

# Set up logging
logging.basicConfig(level=logging.ERROR, filename='errors.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')

# Custom Dataset Class to Load PDF Files
class PDFDataset(Dataset):
    def __init__(self, folder_path, max_length=1000):
        self.pdf_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.pdf')]
        self.data = [self.extract_text_from_pdf(f) for f in self.pdf_files]
        self.max_length = max_length

    def extract_text_from_pdf(self, pdf_path):
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ''
                for page in range(len(reader.pages)):
                    text += reader.pages[page].extract_text()
            return self.preprocess_text(text)
        except Exception as e:
            logging.error(f"Error processing {pdf_path}: {str(e)}")
            return ""

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        return text.strip()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        tensor_data = torch.tensor([ord(c) for c in text], dtype=torch.float32)

        # Padding to ensure all tensors have the same length
        if len(tensor_data) < self.max_length:
            padding = torch.zeros(self.max_length - len(tensor_data), dtype=torch.float32)
            tensor_data = torch.cat((tensor_data, padding), dim=0)
        else:
            tensor_data = tensor_data[:self.max_length]

        return tensor_data, tensor_data

# Define the neural network model
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Training function
def train_model(model, dataloader, num_epochs=40, lr=0.0001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(num_epochs):
        model.train()
        for features, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Function to save the trained model
def save_model(model, path='simple_nn_model.pth'):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

# Function to load the model
def load_model(model, path='simple_nn_model.pth'):
    model.load_state_dict(torch.load(path))
    model.eval()
    print(f"Model loaded from {path}")

# Evaluation function
def evaluate_model(model, dataloader):
    model.eval()
    all_preds = []
    all_targets = []

    for features, targets in dataloader:
        outputs = model(features)
        all_preds.extend(outputs.detach().numpy().flatten())
        all_targets.extend(targets.numpy().flatten())

    y_true = [1 if x > 0.5 else 0 for x in all_targets]
    y_pred = [1 if x > 0.5 else 0 for x in all_preds]

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)

    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-score: {f1:.2f}")
    print(f"Accuracy: {accuracy:.2f}")

# Main function
if __name__ == "__main__":
    folder_path = r"C:\\Users\\JL COMPUTERS\\Desktop\\AiInternTask\\pdfs"

    # Create the dataset
    dataset = PDFDataset(folder_path)

    # Create the dataset
    dataset = PDFDataset(folder_path)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Model parameters
    input_size = 1000  # Adjust based on your text data
    hidden_size = 500
    output_size = 1000  # Adjust based on your text data

    # Create the model
    model = SimpleNN(input_size, hidden_size, output_size)

    # Train the model
    train_model(model, dataloader, num_epochs=40, lr=0.0001)

    # Save the trained model
    save_model(model, path='simple_nn_model.pth')

    # Load the model (for testing)
    load_model(model, path='simple_nn_model.pth')

    # Evaluate the model
    evaluate_model(model, dataloader)
