import os
import torch
import torch.nn as nn
import torch.optim as optim
import PyPDF2
import re
from torch.utils.data import Dataset, DataLoader

# Custom Dataset Class to Load PDF Files
class PDFDataset(Dataset):
    def __init__(self, folder_path):
        self.pdf_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.pdf')]
        self.data = [self.extract_text_from_pdf(f) for f in self.pdf_files]

    def extract_text_from_pdf(self, pdf_path):
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ''
            for page in range(len(reader.pages)):
                text += reader.pages[page].extract_text()
        return self.preprocess_text(text)

    def preprocess_text(self, text):
        # Basic preprocessing: lowercasing and removing non-alphabet characters
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)  # Remove non-alphabet characters
        return text.strip()

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Convert the preprocessed text into tensor (basic character encoding)
        text = self.data[idx]
        # Here we use ASCII encoding for simplicity; adjust this for more complex tasks
        tensor_data = torch.tensor([ord(c) for c in text], dtype=torch.float32)
        return tensor_data[:1000], tensor_data  # Return input and target (for simplicity)

# Define the neural network model
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # First fully connected layer
        self.relu = nn.ReLU()                         # Activation function (ReLU)
        self.fc2 = nn.Linear(hidden_size, output_size) # Output layer

    def forward(self, x):
        # Forward propagation step
        out = self.fc1(x)  # Input -> Hidden layer
        out = self.relu(out)  # Apply ReLU activation
        out = self.fc2(out)  # Hidden -> Output layer
        return out

# Training function
def train_model(model, dataloader, num_epochs=10, lr=0.001):
    # Define the loss function (MSELoss for regression)
    criterion = nn.MSELoss()

    # Define the optimizer (Adam)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        for features, targets in dataloader:
            # Step 1: Forward pass
            outputs = model(features)

            # Step 2: Compute the loss
            loss = criterion(outputs, targets)

            # Step 3: Backward pass (backpropagation)
            optimizer.zero_grad()  # Clear gradients
            loss.backward()  # Backpropagate the error

            # Step 4: Optimize/Update weights
            optimizer.step()

        # Print progress every epoch
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Function to save the trained model
def save_model(model, path='simple_nn_model.pth'):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

# Function to load the model
def load_model(model, path='simple_nn_model.pth'):
    model.load_state_dict(torch.load(path))
    model.eval()  # Set to evaluation mode for inference
    print(f"Model loaded from {path}")

# Main function
if __name__ == "__main__":
    folder_path = r"C:\\Users\\JL COMPUTERS\\Desktop\\AiInternTask\\pdfs"
    
    # Create the dataset
    dataset = PDFDataset(folder_path)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Model parameters
    input_size = 1000  # Fixed input size (adjust based on your text data)
    hidden_size = 50
    output_size = 1000  # Output size should match your tensor data shape

    # Create the model
    model = SimpleNN(input_size, hidden_size, output_size)

    # Train the model
    train_model(model, dataloader, num_epochs=10, lr=0.001)

    # Save the trained model
    save_model(model, path='simple_nn_model.pth')

    # Load the model (for testing)
    load_model(model, path='simple_nn_model.pth')

    # Test the model (with dummy data)
    model.eval()  # Set the model to evaluation mode
    for features, _ in dataloader:
        test_output = model(features)  # Test on first batch
        print("Test output (first batch):\n", test_output)
        break  # Only print the first batch
