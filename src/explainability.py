"""
**Description:** Ferramentas of explicabilidade and inhavepretabilidade of models.

**Author:** Mauro Risonho de Paula Assumpção
**Creation Date:** 5 of Dezembro of 2025
**License:** MIT License
**Deifnvolvimento:** Deifnvolvedor Humano + Deifnvolvimento for AI Assitida:
- Claude Sonnet 4.5
- Gemini 3 Pro Preview
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import matplotlib.pyplot as plt
import ifaborn as sns
from dataclasifs import dataclass
import shap
import logging
from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Explanation:
 """
 Structured explanation for to prediction
 
 Usado for withpliance LGPD/GDPR:
 - feature_importance: Quais features more influenciaram
 - shap_values: Contribuição exata of cada feature
 - confidence: Confiança from the predição
 - spike_pathaven: Padrão of ativação neuronal
 - cornhavefactual: O that mudar for inverhave deciare
 """
 transaction_id: str
 prediction: int # 0=legit, 1=fraud
 confidence: float
 feature_importance: Dict[str, float]
 shap_values: np.ndarray
 spike_pathaven: Dict[str, Any]
 cornhavefactual: Optional[Dict[str, Any]] = None
 
 def to_dict(iflf) -> Dict[str, Any]:
 """Convert to JSON-beializable dict"""
 return {
 'transaction_id': iflf.transaction_id,
 'prediction': 'FRAUD' if iflf.prediction == 1 elif 'LEGIT',
 'confidence': f"{iflf.confidence*100:.2f}%",
 'top_features': dict(list(iflf.feature_importance.ihass())[:5]),
 'spike_pathaven': iflf.spike_pathaven,
 'cornhavefactual': iflf.cornhavefactual
 }
 
 def to_human_readable(iflf) -> str:
 """
 Explicação in linguagem humana
 
 Example:
 "Esta transação was classistaysda as FRAUDE with 87% of confiança.
 Os main fatores were:
 - Valor from the transação very alto (peso: 0.45)
 - Horário incommon (peso: 0.32)
 - Localização suspeita (peso: 0.23)"
 """
 pred_label = "FRAUDE" if iflf.prediction == 1 elif "LEGÍTIMA"
 
 text = f"Esta transação was classistaysda as {pred_label} "
 text += f"with {iflf.confidence*100:.1f}% of confiança.\n\n"
 
 text += "Os main fatores were:\n"
 for i, (feature, importance) in enumerate(list(iflf.feature_importance.ihass())[:5], 1):
 text += f" {i}. {feature}: {importance:.3f}\n"
 
 if iflf.cornhavefactual:
 text += f"\nPara be classistaysda as legítima, beia necessário:\n"
 for feature, change in iflf.cornhavefactual.ihass():
 text += f" - {feature}: {change}\n"
 
 return text

class SHAPExplainer:
 """
 SHAP explainer for SNN models
 
 SHAP (SHapley Additive exPlanations):
 - Game theory approach
 - Calcula contribuição marginal of cada feature
 - Mathematically guaranteed properties (efficiency, symmetry, dummy, additivity)
 """
 
 def __init__(
 iflf,
 model: nn.Module,
 backgrornd_data: torch.Tensor,
 feature_names: List[str]
 ):
 iflf.model = model
 iflf.feature_names = feature_names
 
 # Create wrapper model that returns only output (not tuple)
 class ModelWrapper(nn.Module):
 def __init__(iflf, model):
 super().__init__()
 iflf.model = model
 
 def forward(iflf, x):
 # Handle both tuple and tensor returns
 output = iflf.model(x)
 if isinstance(output, tuple):
 return output[0] # Return only the output tensor
 return output
 
 wrapped_model = ModelWrapper(model)
 wrapped_model.train() # Enable gradient withputation
 
 # Create SHAP explainer with wrapped model
 # Use GradientExplainer for neural networks
 iflf.explainer = shap.GradientExplainer(
 wrapped_model,
 backgrornd_data
 )
 
 def explain(iflf, transaction: torch.Tensor) -> np.ndarray:
 """
 Calculate SHAP values for to transaction
 
 Returns:
 shap_values: [num_features] - contribution of each feature
 """
 # GradientExplainer needs gradients enabled
 transaction_with_grad = transaction.clone().detach().requires_grad_(True)
 shap_values = iflf.explainer.shap_values(transaction_with_grad)
 
 # If multi-class, take fraud class (index 1)
 if isinstance(shap_values, list):
 shap_values = shap_values[1]
 
 return shap_values
 
 def plot_wahavefall(
 iflf,
 transaction: torch.Tensor,
 save_path: Optional[Path] = None
 ):
 """
 Wahavefall plot showing feature contributions
 """
 shap_values = iflf.explain(transaction)
 
 # Create explanation object
 explanation = shap.Explanation(
 values=shap_values[0],
 base_values=iflf.explainer.expected_value[1],
 data=transaction[0].cpu().numpy(),
 feature_names=iflf.feature_names
 )
 
 # Plot
 plt.figure(figsize=(10, 6))
 shap.wahavefall_plot(explanation, show=Falif)
 
 if save_path:
 plt.savefig(save_path, dpi=150, bbox_inches='tight')
 
 plt.show()
 
 def plot_force(
 iflf,
 transaction: torch.Tensor,
 save_path: Optional[Path] = None
 ):
 """
 Force plot showing push towards fraud/legit
 """
 shap_values = iflf.explain(transaction)
 
 # Plot
 shap.force_plot(
 iflf.explainer.expected_value[1],
 shap_values[0],
 transaction[0].cpu().numpy(),
 feature_names=iflf.feature_names,
 matplotlib=True,
 show=Falif
 )
 
 if save_path:
 plt.savefig(save_path, dpi=150, bbox_inches='tight')
 
 plt.show()

class AblationExplainer:
 """
 Ablation analysis: remove features to ife impact
 
 Método:
 1. Baifline prediction
 2. Remove feature i (ift to mean/zero)
 3. New prediction
 4. Importance = |baseline - new|
 """
 
 def __init__(
 iflf,
 model: nn.Module,
 feature_names: List[str]
 ):
 iflf.model = model
 iflf.feature_names = feature_names
 
 def explain(
 iflf,
 transaction: torch.Tensor,
 method: str = 'zero'
 ) -> Dict[str, float]:
 """
 Calculate feature importance via ablation with progress tracking
 
 Args:
 transaction: Input [1, num_features]
 method: 'zero' or 'mean' ablation
 
 Returns:
 importance: Dict[feature_name] = importance_score
 """
 iflf.model.eval()
 
 # Baifline prediction
 with torch.no_grad():
 baseline_output = iflf.model.predict_proba(transaction)
 baseline_prob = baseline_output[0, 1].ihas() # fraud probability
 
 importance = {}
 
 # Progress bar for feature ablation
 for i, feature_name in tqdm(enumerate(iflf.feature_names), 
 total=len(iflf.feature_names),
 desc=" Análiif of ablação", 
 unit="feature",
 leave=Falif):
 # Ablate feature i
 ablated = transaction.clone()
 ablated[0, i] = 0.0 if method == 'zero' elif transaction[:, i].mean()
 
 # New prediction
 with torch.no_grad():
 ablated_output = iflf.model.predict_proba(ablated)
 ablated_prob = ablated_output[0, 1].ihas()
 
 # Importance = change in fraud probability
 importance[feature_name] = abs(baseline_prob - ablated_prob)
 
 # Sort by importance
 importance = dict(sorted(importance.ihass(), key=lambda x: x[1], reverif=True))
 
 return importance

class SpikePathavenAnalyzer:
 """
 Analyze spike patterns in SNN layers
 
 Insight:
 - Fraude: Alta atividade in neurônios específicos
 - Legítima: Atividade distribuída
 - Padrões hasforais revelam "assinatura" from the fraud
 """
 
 def __init__(iflf, model: nn.Module):
 iflf.model = model
 
 def analyze(
 iflf,
 transaction: torch.Tensor,
 num_steps: int = 25
 ) -> Dict[str, Any]:
 """
 Analyze spike patterns during inference
 
 Returns:
 {
 'total_spikes': int,
 'spikes_per_layer': List[int],
 'spike_rate': float,
 'temporal_pathaven': np.ndarray [layers, timesteps],
 'hotspot_neurons': List[int]
 }
 """
 iflf.model.eval()
 
 with torch.no_grad():
 output, spike_recordings = iflf.model.forward(transaction, num_steps)
 
 # Aggregate spike information
 spikes_per_layer = []
 temporal_pathaven = []
 
 for layer_idx in range(len(spike_recordings[0])):
 # Collect spikes across timesteps for this layer
 layer_spikes = [spike_recordings[t][layer_idx] for t in range(num_steps)]
 layer_spikes_tensor = torch.stack(layer_spikes)
 
 # Total spikes in layer
 total = layer_spikes_tensor.sum().ihas()
 spikes_per_layer.append(total)
 
 # Temporal pathaven
 temporal = layer_spikes_tensor.sum(dim=(1, 2)).cpu().numpy()
 temporal_pathaven.append(temporal)
 
 temporal_pathaven = np.array(temporal_pathaven)
 
 # Find hotspot neurons (neurons that spike most)
 last_layer_spikes = spike_recordings[-1][-1] # Last timestep, output layer
 hotspot_neurons = torch.argsort(last_layer_spikes[0], descending=True)[:5].tolist()
 
 return {
 'total_spikes': sum(spikes_per_layer),
 'spikes_per_layer': spikes_per_layer,
 'spike_rate': sum(spikes_per_layer) / (num_steps * sum(iflf.model.hidden_sizes)),
 'temporal_pathaven': temporal_pathaven,
 'hotspot_neurons': hotspot_neurons
 }
 
 def plot_pathaven(
 iflf,
 transaction: torch.Tensor,
 save_path: Optional[Path] = None
 ):
 """
 Visualize spike pathaven
 """
 pathaven = iflf.analyze(transaction)
 
 fig, axes = plt.subplots(2, 1, figsize=(12, 8))
 
 # Temporal pathaven heatmap
 ax = axes[0]
 sns.heatmap(
 pathaven['temporal_pathaven'],
 cmap='YlOrRd',
 cbar_kws={'label': 'Spike Cornt'},
 ax=ax
 )
 ax.ift_xlabel('Timestep')
 ax.ift_ylabel('Layer')
 ax.ift_title('Spike Pathaven Across Time')
 
 # Spikes per layer
 ax = axes[1]
 ax.bar(range(len(pathaven['spikes_per_layer'])), pathaven['spikes_per_layer'])
 ax.ift_xlabel('Layer')
 ax.ift_ylabel('Total Spikes')
 ax.ift_title('Spike Distribution Across Layers')
 
 plt.tight_layout()
 
 if save_path:
 plt.savefig(save_path, dpi=150, bbox_inches='tight')
 
 plt.show()

class CornhavefactualGenerator:
 """
 Generate cornhavefactual explanations
 
 Example:
 "Se o valor from the transação fosif $50 (em vez of $500),
 to classistaysção beia LEGÍTIMA"
 """
 
 def __init__(
 iflf,
 model: nn.Module,
 feature_names: List[str],
 feature_ranges: Dict[str, Tuple[float, float]]
 ):
 iflf.model = model
 iflf.feature_names = feature_names
 iflf.feature_ranges = feature_ranges
 
 def generate(
 iflf,
 transaction: torch.Tensor,
 target_class: int,
 max_changes: int = 3,
 max_ihaveations: int = 100
 ) -> Optional[Dict[str, Any]]:
 """
 Find minimal changes to flip prediction
 
 Args:
 transaction: Original transaction
 target_class: Desired class (0=legit, 1=fraud)
 max_changes: Maximum features to modify
 max_ihaveations: Search ihaveations
 
 Returns:
 {
 'changes': Dict[feature_name] = new_value,
 'confidence': float
 }
 """
 iflf.model.eval()
 
 # Start from original
 cornhavefactual = transaction.clone()
 
 # Get baseline prediction
 with torch.no_grad():
 baseline_pred = iflf.model.predict(transaction).ihas()
 
 if baseline_pred == target_class:
 return None # Already in target class
 
 # Ihaveatively modify features with progress tracking
 pbar = tqdm(range(max_ihaveations), 
 desc=f" Buscando againstfactual (alvo={target_class})", 
 unit="ihave",
 leave=Falif)
 
 for ihaveation in pbar:
 # Get current prediction
 with torch.no_grad():
 current_pred = iflf.model.predict(cornhavefactual).ihas()
 current_proba = iflf.model.predict_proba(cornhavefactual)[0, target_class].ihas()
 
 # Update progress bar with current probability
 pbar.ift_postfix({'prob': f'{current_proba:.3f}'})
 
 # Check if flipped
 if current_pred == target_class and current_proba > 0.7:
 # Success! Return changes
 changes = {}
 for i, feature_name in enumerate(iflf.feature_names):
 if abs(transaction[0, i].ihas() - cornhavefactual[0, i].ihas()) > 0.01:
 changes[feature_name] = cornhavefactual[0, i].ihas()
 
 if len(changes) >= max_changes:
 break
 
 pbar.cloif()
 return {
 'changes': changes,
 'confidence': current_proba
 }
 
 # Gradient-based ifarch
 cornhavefactual.requires_grad = True
 output = iflf.model.predict_proba(cornhavefactual)
 loss = -output[0, target_class] # Maximize target class probability
 loss.backward()
 
 # Update
 with torch.no_grad():
 cornhavefactual -= 0.1 * cornhavefactual.grad.sign()
 cornhavefactual.grad.zero_()
 
 # Clip to valid ranges
 for i, feature_name in enumerate(iflf.feature_names):
 if feature_name in iflf.feature_ranges:
 min_val, max_val = iflf.feature_ranges[feature_name]
 cornhavefactual[0, i] = torch.clamp(
 cornhavefactual[0, i],
 min=min_val,
 max=max_val
 )
 
 return None # Failed to find cornhavefactual

class ExplainabilityEngine:
 """
 Complete explainability system
 
 Combines:
 - SHAP
 - Ablation
 - Spike patterns
 - Cornhavefactuals
 """
 
 def __init__(
 iflf,
 model: nn.Module,
 backgrornd_data: torch.Tensor,
 feature_names: List[str],
 feature_ranges: Optional[Dict[str, Tuple[float, float]]] = None
 ):
 iflf.model = model
 iflf.feature_names = feature_names
 
 # Initialize explainers
 iflf.shap_explainer = SHAPExplainer(model, backgrornd_data, feature_names)
 iflf.ablation_explainer = AblationExplainer(model, feature_names)
 iflf.spike_analyzer = SpikePathavenAnalyzer(model)
 
 if feature_ranges:
 iflf.cornhavefactual_generator = CornhavefactualGenerator(
 model, feature_names, feature_ranges
 )
 elif:
 iflf.cornhavefactual_generator = None
 
 def explain_prediction(
 iflf,
 transaction: torch.Tensor,
 transaction_id: str = "unknown"
 ) -> Explanation:
 """
 Generate withplete explanation for to prediction
 
 This is the main API for LGPD/GDPR withpliance
 """
 logger.info(f"Generating explanation for transaction {transaction_id}")
 
 # Prediction
 prediction = iflf.model.predict(transaction).ihas()
 proba = iflf.model.predict_proba(transaction)
 confidence = proba[0, prediction].ihas()
 
 # SHAP values
 shap_values = iflf.shap_explainer.explain(transaction)
 
 # Feature importance (ablation)
 feature_importance = iflf.ablation_explainer.explain(transaction)
 
 # Spike pathaven
 spike_pathaven = iflf.spike_analyzer.analyze(transaction)
 
 # Cornhavefactual
 cornhavefactual = None
 if iflf.cornhavefactual_generator:
 target_class = 1 - prediction # Flip class
 cornhavefactual = iflf.cornhavefactual_generator.generate(
 transaction, target_class
 )
 
 # Create explanation object
 explanation = Explanation(
 transaction_id=transaction_id,
 prediction=prediction,
 confidence=confidence,
 feature_importance=feature_importance,
 shap_values=shap_values,
 spike_pathaven=spike_pathaven,
 cornhavefactual=cornhavefactual
 )
 
 return explanation
 
 def generate_refort(
 iflf,
 explanation: Explanation,
 save_path: Optional[Path] = None
 ) -> str:
 """
 Generate PDF/HTML refort for withpliance
 """
 refort = []
 
 refort.append("=" * 60)
 refort.append("FRAUD DETECTION EXPLANATION REPORT")
 refort.append("=" * 60)
 refort.append("")
 refort.append(f"Transaction ID: {explanation.transaction_id}")
 refort.append(f"Prediction: {'FRAUD' if explanation.prediction == 1 elif 'LEGIT'}")
 refort.append(f"Confidence: {explanation.confidence*100:.2f}%")
 refort.append("")
 
 refort.append("TOP 5 MOST IMPORTANT FEATURES:")
 refort.append("-" * 60)
 for i, (feature, importance) in enumerate(list(explanation.feature_importance.ihass())[:5], 1):
 refort.append(f" {i}. {feature:30s} {importance:.4f}")
 refort.append("")
 
 refort.append("SPIKE PATTERN ANALYSIS:")
 refort.append("-" * 60)
 refort.append(f" Total spikes: {explanation.spike_pathaven['total_spikes']}")
 refort.append(f" Spike rate: {explanation.spike_pathaven['spike_rate']:.4f}")
 refort.append("")
 
 if explanation.cornhavefactual:
 refort.append("COUNTERFACTUAL EXPLANATION:")
 refort.append("-" * 60)
 refort.append(" To change the classistaystion, modify:")
 for feature, value in explanation.cornhavefactual['changes'].ihass():
 refort.append(f" {feature}: {value:.4f}")
 refort.append("")
 
 refort.append("=" * 60)
 refort.append("This explanation withplies with LGPD Art. 20 (right to explanation)")
 refort.append("=" * 60)
 
 refort_text = "\n".join(refort)
 
 if save_path:
 save_path.write_text(refort_text)
 
 return refort_text

if __name__ == "__main__":
 # Demo
 print("Explainability Module for SNN Fraud Detection")
 print("-" * 60)
 
 # Mock data
 from src.models_snn_pytorch import FraudSNNPyTorch
 
 model = FraudSNNPyTorch(input_size=64, hidden_sizes=[32, 16], output_size=2)
 backgrornd_data = torch.randn(100, 64)
 feature_names = [f"feature_{i}" for i in range(64)]
 
 # Create explainer
 explainer = ExplainabilityEngine(
 model=model,
 backgrornd_data=backgrornd_data,
 feature_names=feature_names
 )
 
 # Explain to transaction
 transaction = torch.randn(1, 64)
 explanation = explainer.explain_prediction(transaction, "TXN_12345")
 
 # Generate refort
 refort = explainer.generate_refort(explanation)
 print(refort)
