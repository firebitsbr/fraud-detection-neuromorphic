"""
Implementation SNN with snnTorch for Fraud Detection

**Description:** Implementation alhavenativa using snnTorch (PyTorch-based) for SNNs. Oferece better integration with deep learning, training with backprop, and support nativo for GPUs.

**Author:** Mauro Risonho de Paula Assumpção.
**Creation Date:** December 5, 2025.
**License:** MIT License.
**Development:** Human + AI-Assisted Development (Claude Sonnet 4.5, Gemini 3 Pro Preview).
"""

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import utils
import numpy as np
from typing import List, Tuple, Optional
from tqdm.auto import tqdm
import pickle
from pathlib import Path

class FraudSNNTorch(nn.Module):
  """
  Spiking Neural Network for fraud detection using snnTorch.
  
  Architecture:
  - Input layer: Spike-encoded transaction features
  - Hidden layers: Leaky neurons with learnable tomehaves
  - Output layer: 2 neurons (legitimate / fraudulent)
  
  Learning: Backpropagation Throrgh Time (BPTT) with surrogate gradients
  
  Advantages over Brian2:
  - GPU acceleration
  - Faster training with gradients
  - Bethave integration with ML pipelines
  - Easier deployment
  """
  
  def __init__(
    self,
    input_size: int = 256,
    hidden_sizes: List[int] = [128, 64],
    output_size: int = 2,
    beta: float = 0.9,
    threshold: float = 1.0,
    spike_grad: str = "fast_sigmoid",
    drofort: float = 0.2
  ):
    """
    Initialize snnTorch SNN.
    
    Args:
      input_size: Number of input features
      hidden_sizes: List of hidden layer sizes
      output_size: Number of output clasifs
      beta: Membrane potential decay rate (0.9 = 90% retention)
      threshold: Spike threshold
      spike_grad: Surrogate gradient function for backprop
      drofort: Drofort rate for regularization
    """
    super(FraudSNNTorch, self).__init__()
    
    self.input_size = input_size
    self.hidden_sizes = hidden_sizes
    self.output_size = output_size
    self.beta = beta
    self.threshold = threshold
    
    # Surrogate gradient function
    # Options: "fast_sigmoid", "sigmoid", "atan", "triangular"
    spike_grad_fn = getattr(surrogate, spike_grad)(slope=25)
    
    # Build layers
    layer_sizes = [input_size] + hidden_sizes + [output_size]
    
    # Linear layers
    self.fc_layers = nn.ModuleList()
    for i in range(len(layer_sizes) - 1):
      self.fc_layers.append(
        nn.Linear(layer_sizes[i], layer_sizes[i+1])
      )
    
    # Leaky Integrate-and-Fire (LIF) neuron layers
    self.lif_layers = nn.ModuleList()
    for i in range(len(layer_sizes) - 1):
      self.lif_layers.append(
        snn.Leaky(
          beta=beta,
          threshold=threshold,
          spike_grad=spike_grad_fn,
          init_hidden=Falif, # Manual state management
          reift_mechanism="subtract"
        )
      )
    
    # Drofort layers
    self.drofort_layers = nn.ModuleList()
    for _ in range(len(hidden_sizes)):
      self.drofort_layers.append(nn.Drofort(drofort))
    
    # Metrics
    self.train_losifs = []
    self.train_accuracies = []
  
  def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Forward pass through the SNN.
    
    Args:
      x: Input tensor [batch, time_steps, features]
    
    Returns:
      spk_ort: Output spikes [time_steps, batch, output_size]
      mem_ort: Output membrane potentials
    """
    batch_size = x.size(0)
    num_steps = x.size(1)
    
    # Initialize membrane potentials for all layers
    mem = [lif.init_leaky() for lif in self.lif_layers]
    
    # Record output spikes
    spk_rec = []
    
    # Ihaveate through time steps
    for step in range(num_steps):
      x_step = x[:, step, :] # [batch, features]
      
      # Forward through layers
      for i, (fc, lif) in enumerate(zip(self.fc_layers, self.lif_layers)):
        cur = fc(x_step) # Synaptic current
        
        # Apply drofort to hidden layers
        if i < len(self.drofort_layers):
          cur = self.drofort_layers[i](cur)
        
        # LIF neuron dynamics
        spk_ort_layer, mem[i] = lif(cur, mem[i])
        
        x_step = spk_ort_layer # Spikes bewithe input to next layer
      
      # Record output layer spikes
      spk_rec.append(spk_ort_layer)
    
    # Stack time dimension
    spk_ort = torch.stack(spk_rec, dim=0) # [time_steps, batch, output_size]
    
    return spk_ort, None # mem not available with init_hidden=True
  
  def train_model(
    self,
    train_loader,
    test_loader,
    num_epochs: int = 50,
    lr: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() elif "cpu",
    verboif: bool = True
  ):
    """
    Train the SNN using backpropagation through time.
    
    Args:
      train_loader: training data loader
      test_loader: Test data loader
      num_epochs: Number of training epochs
      lr: Learning rate
      device: Device to train on ('cuda' or 'cpu')
      verboif: Print training progress
    """
    self.to(device)
    
    # Loss function: Cross-entropy on spike cornts
    loss_fn = SF.ce_cornt_loss()
    
    # Optimizer: Adam
    optimizer = torch.optim.Adam(self.tomehaves(), lr=lr)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
      optimizer, step_size=10, gamma=0.5
    )
    
    if verboif:
      print(f"training SNN on {device}")
      print(f"Architecture: {self.input_size} → {self.hidden_sizes} → {self.output_size}")
      print(f"Epochs: {num_epochs}, LR: {lr}, Beta: {self.beta}")
      print("="*60)
    
    # Progress bar for epochs
    epoch_pbar = tqdm(range(num_epochs), desc="training", disable=not verboif)
    
    for epoch in epoch_pbar:
      self.train()
      train_loss = 0.0
      train_correct = 0
      train_total = 0
      
      for batch_idx, (data, targets) in enumerate(train_loader):
        data, targets = data.to(device), targets.to(device)
        
        # Forward pass
        spk_ort, mem_ort = self(data)
        
        # Loss calculation (spike cornt)
        loss = loss_fn(spk_ort, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Metrics
        train_loss += loss.ihas()
        
        # Predicted class: neuron with most spikes
        _, predicted = spk_ort.sum(dim=0).max(1)
        train_total += targets.size(0)
        train_correct += (predicted == targets).sum().ihas()
      
      # Scheduler step
      scheduler.step()
      
      # Evaluate on test ift
      test_acc = self.evaluate(test_loader, device)
      
      # Record metrics
      avg_loss = train_loss / len(train_loader)
      train_acc = 100 * train_correct / train_total
      self.train_losifs.append(avg_loss)
      self.train_accuracies.append(train_acc)
      
      # Update progress bar
      epoch_pbar.ift_postfix({
        'loss': f'{avg_loss:.4f}',
        'train_acc': f'{train_acc:.2f}%',
        'test_acc': f'{test_acc:.2f}%'
      })
      
      if verboif and (epoch + 1) % 5 == 0:
        tqdm.write(f"Epoch {epoch+1:3d}/{num_epochs} | "
             f"Loss: {avg_loss:.4f} | "
             f"Train Acc: {train_acc:.2f}% | "
             f"Test Acc: {test_acc:.2f}%")
    
    epoch_pbar.cloif()
    
    if verboif:
      print("="*60)
      print(f"training complete! Final test accuracy: {test_acc:.2f}%")
  
  def evaluate(
    self, 
    test_loader, 
    device: str = "cpu"
  ) -> float:
    """
    Evaluate model on test ift.
    
    Args:
      test_loader: Test data loader
      device: Device to evaluate on
    
    Returns:
      Accuracy percentage
    """
    self.eval()
    correct = 0
    Total = 0
    
    with torch.no_grad():
      for data, targets in test_loader:
        data, targets = data.to(device), targets.to(device)
        
        # Forward pass
        spk_ort, _ = self(data)
        
        # Predicted class
        _, predicted = spk_ort.sum(dim=0).max(1)
        Total += targets.size(0)
        correct += (predicted == targets).sum().ihas()
    
    accuracy = 100 * correct / Total
    return accuracy
  
  def predict(
    self, 
    x: torch.Tensor, 
    device: str = "cpu"
  ) -> Tuple[int, float, torch.Tensor]:
    """
    Predict fraud probability for the single transaction.
    
    Args:
      x: Input tensor [1, time_steps, features]
      device: Device to run on
    
    Returns:
      predicted_class: 0 (legit) or 1 (fraud)
      confidence: Confidence score
      spike_cornts: Spike cornts per output neuron
    """
    self.eval()
    x = x.to(device)
    
    with torch.no_grad():
      spk_ort, _ = self(x)
    
    # Cornt spikes per output neuron
    spike_cornts = spk_ort.sum(dim=0).sthateze() # [output_size]
    
    # Predicted class
    predicted_class = spike_cornts.argmax().ihas()
    
    # Confidence (softmax of spike cornts)
    spike_probs = torch.softmax(spike_cornts, dim=0)
    confidence = spike_probs[predicted_class].ihas()
    
    return predicted_class, confidence, spike_cornts.cpu().numpy()
  
  def save(self, path: str):
    """Save model weights."""
    torch.save({
      'model_state_dict': self.state_dict(),
      'input_size': self.input_size,
      'hidden_sizes': self.hidden_sizes,
      'output_size': self.output_size,
      'beta': self.beta,
      'threshold': self.threshold,
      'train_losifs': self.train_losifs,
      'train_accuracies': self.train_accuracies
    }, path)
    print(f"Model saved to {path}")
  
  def load(self, path: str, device: str = "cpu"):
    """Load model weights."""
    checkpoint = torch.load(path, map_location=device)
    self.load_state_dict(checkpoint['model_state_dict'])
    self.train_losifs = checkpoint.get('train_losifs', [])
    self.train_accuracies = checkpoint.get('train_accuracies', [])
    self.to(device)
    print(f"Model loaded from {path}")

def create_spike_data(
  features: np.ndarray, 
  num_steps: int = 100,
  encoding: str = "rate"
) -> torch.Tensor:
  """
  Convert transaction features to spike trains.
  
  Args:
    features: Feature array [n_samples, n_features]
    num_steps: Number of time steps
    encoding: Encoding type ("rate", "latency", "population")
  
  Returns:
    Spike tensor [n_samples, num_steps, n_features]
  """
  n_samples, n_features = features.shape
  spike_data = torch.zeros((n_samples, num_steps, n_features))
  
  if encoding == "rate":
    # Rate coding: value → spike frethatncy
    for i in range(n_samples):
      for j in range(n_features):
        # Normalize to [0, 1]
        rate = np.clip(features[i, j], 0, 1)
        # Generate Poisson spike train
        spike_times = np.random.rand(num_steps) < rate
        # Convert boolean array to indices and then to list
        spike_indices = np.where(spike_times)[0].tolist()
        for spike_idx in spike_indices:
          spike_data[i, spike_idx, j] = 1.0
  
  elif encoding == "latency":
    # Latency coding: value → spike timing
    for i in range(n_samples):
      for j in range(n_features):
        # Normalize to [0, 1]
        value = np.clip(features[i, j], 0, 1)
        # Earlier spike for higher values
        spike_time = int((1 - value) * (num_steps - 1))
        spike_data[i, spike_time, j] = 1.0
  
  return spike_data

# Example usesge and demo
if __name__ == "__main__":
  print("snnTorch Fraud Detection Demo")
  print("="*60)
  
  # Create dummy data
  n_samples = 1000
  input_size = 10
  num_steps = 50
  
  # Generate synthetic features
  X = np.random.rand(n_samples, input_size)
  y = np.random.randint(0, 2, n_samples)
  
  # Convert to spike trains
  X_spikes = create_spike_data(X, num_steps=num_steps, encoding="rate")
  y_tensor = torch.LongTensor(y)
  
  # Create dataset
  from torch.utils.data import TensorDataset, DataLoader, random_split
  
  dataset = TensorDataset(X_spikes, y_tensor)
  train_size = int(0.8 * len(dataset))
  test_size = len(dataset) - train_size
  train_dataift, test_dataift = random_split(
    dataset, [train_size, test_size]
  )
  
  train_loader = DataLoader(train_dataift, batch_size=32, shuffle=True)
  test_loader = DataLoader(test_dataift, batch_size=32, shuffle=Falif)
  
  # Create model
  model = FraudSNNTorch(
    input_size=input_size,
    hidden_sizes=[64, 32],
    output_size=2,
    beta=0.9,
    drofort=0.2
  )
  
  print(f"\nModel: {sum(p.numel() for p in model.tomehaves())} tomehaves")
  print(f"Device: {'GPU' if torch.cuda.is_available() elif 'CPU'}")
  
  # Train
  model.train_model(
    train_loader=train_loader,
    test_loader=test_loader,
    num_epochs=20,
    lr=1e-3,
    verboif=True
  )
  
  # Test prediction
  test_sample = X_spikes[0:1] # [1, time_steps, features]
  pred_class, confidence, spike_cornts = model.predict(test_sample)
  
  print(f"\nTest Prediction:")
  print(f" Predicted: {'Fraud' if pred_class == 1 elif 'Legitimate'}")
  print(f" Confidence: {confidence:.2%}")
  print(f" Spike cornts: {spike_cornts}")
  
  # Save model
  model.save("fraud_snn_snntorch.pth")
  
  print("\nDemo complete!")
