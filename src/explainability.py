"""
**Descrição:** Ferramentas de explicabilidade e interpretabilidade de modelos.

**Autor:** Mauro Risonho de Paula Assumpção
**Data de Criação:** 5 de Dezembro de 2025
**Licença:** MIT License
**Desenvolvimento:** Desenvolvedor Humano + Desenvolvimento por AI Assitida:
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
import seaborn as sns
from dataclasses import dataclass
import shap
import logging
from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Explanation:
 """
 Structured explanation for a prediction
 
 Usado para compliance LGPD/GDPR:
 - feature_importance: Quais features mais influenciaram
 - shap_values: Contribuição exata de cada feature
 - confidence: Confiança da predição
 - spike_pattern: Padrão de ativação neuronal
 - counterfactual: O que mudar para inverter decisão
 """
 transaction_id: str
 prediction: int # 0=legit, 1=fraud
 confidence: float
 feature_importance: Dict[str, float]
 shap_values: np.ndarray
 spike_pattern: Dict[str, Any]
 counterfactual: Optional[Dict[str, Any]] = None
 
 def to_dict(self) -> Dict[str, Any]:
 """Convert to JSON-serializable dict"""
 return {
 'transaction_id': self.transaction_id,
 'prediction': 'FRAUD' if self.prediction == 1 else 'LEGIT',
 'confidence': f"{self.confidence*100:.2f}%",
 'top_features': dict(list(self.feature_importance.items())[:5]),
 'spike_pattern': self.spike_pattern,
 'counterfactual': self.counterfactual
 }
 
 def to_human_readable(self) -> str:
 """
 Explicação em linguagem humana
 
 Exemplo:
 "Esta transação foi classificada como FRAUDE com 87% de confiança.
 Os principais fatores foram:
 - Valor da transação muito alto (peso: 0.45)
 - Horário incomum (peso: 0.32)
 - Localização suspeita (peso: 0.23)"
 """
 pred_label = "FRAUDE" if self.prediction == 1 else "LEGÍTIMA"
 
 text = f"Esta transação foi classificada como {pred_label} "
 text += f"com {self.confidence*100:.1f}% de confiança.\n\n"
 
 text += "Os principais fatores foram:\n"
 for i, (feature, importance) in enumerate(list(self.feature_importance.items())[:5], 1):
 text += f" {i}. {feature}: {importance:.3f}\n"
 
 if self.counterfactual:
 text += f"\nPara ser classificada como legítima, seria necessário:\n"
 for feature, change in self.counterfactual.items():
 text += f" - {feature}: {change}\n"
 
 return text

class SHAPExplainer:
 """
 SHAP explainer for SNN models
 
 SHAP (SHapley Additive exPlanations):
 - Game theory approach
 - Calcula contribuição marginal de cada feature
 - Mathematically guaranteed properties (efficiency, symmetry, dummy, additivity)
 """
 
 def __init__(
 self,
 model: nn.Module,
 background_data: torch.Tensor,
 feature_names: List[str]
 ):
 self.model = model
 self.feature_names = feature_names
 
 # Create wrapper model that returns only output (not tuple)
 class ModelWrapper(nn.Module):
 def __init__(self, model):
 super().__init__()
 self.model = model
 
 def forward(self, x):
 # Handle both tuple and tensor returns
 output = self.model(x)
 if isinstance(output, tuple):
 return output[0] # Return only the output tensor
 return output
 
 wrapped_model = ModelWrapper(model)
 wrapped_model.train() # Enable gradient computation
 
 # Create SHAP explainer with wrapped model
 # Use GradientExplainer for neural networks
 self.explainer = shap.GradientExplainer(
 wrapped_model,
 background_data
 )
 
 def explain(self, transaction: torch.Tensor) -> np.ndarray:
 """
 Calculate SHAP values for a transaction
 
 Returns:
 shap_values: [num_features] - contribution of each feature
 """
 # GradientExplainer needs gradients enabled
 transaction_with_grad = transaction.clone().detach().requires_grad_(True)
 shap_values = self.explainer.shap_values(transaction_with_grad)
 
 # If multi-class, take fraud class (index 1)
 if isinstance(shap_values, list):
 shap_values = shap_values[1]
 
 return shap_values
 
 def plot_waterfall(
 self,
 transaction: torch.Tensor,
 save_path: Optional[Path] = None
 ):
 """
 Waterfall plot showing feature contributions
 """
 shap_values = self.explain(transaction)
 
 # Create explanation object
 explanation = shap.Explanation(
 values=shap_values[0],
 base_values=self.explainer.expected_value[1],
 data=transaction[0].cpu().numpy(),
 feature_names=self.feature_names
 )
 
 # Plot
 plt.figure(figsize=(10, 6))
 shap.waterfall_plot(explanation, show=False)
 
 if save_path:
 plt.savefig(save_path, dpi=150, bbox_inches='tight')
 
 plt.show()
 
 def plot_force(
 self,
 transaction: torch.Tensor,
 save_path: Optional[Path] = None
 ):
 """
 Force plot showing push towards fraud/legit
 """
 shap_values = self.explain(transaction)
 
 # Plot
 shap.force_plot(
 self.explainer.expected_value[1],
 shap_values[0],
 transaction[0].cpu().numpy(),
 feature_names=self.feature_names,
 matplotlib=True,
 show=False
 )
 
 if save_path:
 plt.savefig(save_path, dpi=150, bbox_inches='tight')
 
 plt.show()

class AblationExplainer:
 """
 Ablation analysis: remove features to see impact
 
 Método:
 1. Baseline prediction
 2. Remove feature i (set to mean/zero)
 3. New prediction
 4. Importance = |baseline - new|
 """
 
 def __init__(
 self,
 model: nn.Module,
 feature_names: List[str]
 ):
 self.model = model
 self.feature_names = feature_names
 
 def explain(
 self,
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
 self.model.eval()
 
 # Baseline prediction
 with torch.no_grad():
 baseline_output = self.model.predict_proba(transaction)
 baseline_prob = baseline_output[0, 1].item() # fraud probability
 
 importance = {}
 
 # Progress bar for feature ablation
 for i, feature_name in tqdm(enumerate(self.feature_names), 
 total=len(self.feature_names),
 desc=" Análise de ablação", 
 unit="feature",
 leave=False):
 # Ablate feature i
 ablated = transaction.clone()
 ablated[0, i] = 0.0 if method == 'zero' else transaction[:, i].mean()
 
 # New prediction
 with torch.no_grad():
 ablated_output = self.model.predict_proba(ablated)
 ablated_prob = ablated_output[0, 1].item()
 
 # Importance = change in fraud probability
 importance[feature_name] = abs(baseline_prob - ablated_prob)
 
 # Sort by importance
 importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
 
 return importance

class SpikePatternAnalyzer:
 """
 Analyze spike patterns in SNN layers
 
 Insight:
 - Fraude: Alta atividade em neurônios específicos
 - Legítima: Atividade distribuída
 - Padrões temporais revelam "assinatura" da fraude
 """
 
 def __init__(self, model: nn.Module):
 self.model = model
 
 def analyze(
 self,
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
 'temporal_pattern': np.ndarray [layers, timesteps],
 'hotspot_neurons': List[int]
 }
 """
 self.model.eval()
 
 with torch.no_grad():
 output, spike_recordings = self.model.forward(transaction, num_steps)
 
 # Aggregate spike information
 spikes_per_layer = []
 temporal_pattern = []
 
 for layer_idx in range(len(spike_recordings[0])):
 # Collect spikes across timesteps for this layer
 layer_spikes = [spike_recordings[t][layer_idx] for t in range(num_steps)]
 layer_spikes_tensor = torch.stack(layer_spikes)
 
 # Total spikes in layer
 total = layer_spikes_tensor.sum().item()
 spikes_per_layer.append(total)
 
 # Temporal pattern
 temporal = layer_spikes_tensor.sum(dim=(1, 2)).cpu().numpy()
 temporal_pattern.append(temporal)
 
 temporal_pattern = np.array(temporal_pattern)
 
 # Find hotspot neurons (neurons that spike most)
 last_layer_spikes = spike_recordings[-1][-1] # Last timestep, output layer
 hotspot_neurons = torch.argsort(last_layer_spikes[0], descending=True)[:5].tolist()
 
 return {
 'total_spikes': sum(spikes_per_layer),
 'spikes_per_layer': spikes_per_layer,
 'spike_rate': sum(spikes_per_layer) / (num_steps * sum(self.model.hidden_sizes)),
 'temporal_pattern': temporal_pattern,
 'hotspot_neurons': hotspot_neurons
 }
 
 def plot_pattern(
 self,
 transaction: torch.Tensor,
 save_path: Optional[Path] = None
 ):
 """
 Visualize spike pattern
 """
 pattern = self.analyze(transaction)
 
 fig, axes = plt.subplots(2, 1, figsize=(12, 8))
 
 # Temporal pattern heatmap
 ax = axes[0]
 sns.heatmap(
 pattern['temporal_pattern'],
 cmap='YlOrRd',
 cbar_kws={'label': 'Spike Count'},
 ax=ax
 )
 ax.set_xlabel('Timestep')
 ax.set_ylabel('Layer')
 ax.set_title('Spike Pattern Across Time')
 
 # Spikes per layer
 ax = axes[1]
 ax.bar(range(len(pattern['spikes_per_layer'])), pattern['spikes_per_layer'])
 ax.set_xlabel('Layer')
 ax.set_ylabel('Total Spikes')
 ax.set_title('Spike Distribution Across Layers')
 
 plt.tight_layout()
 
 if save_path:
 plt.savefig(save_path, dpi=150, bbox_inches='tight')
 
 plt.show()

class CounterfactualGenerator:
 """
 Generate counterfactual explanations
 
 Exemplo:
 "Se o valor da transação fosse $50 (em vez de $500),
 a classificação seria LEGÍTIMA"
 """
 
 def __init__(
 self,
 model: nn.Module,
 feature_names: List[str],
 feature_ranges: Dict[str, Tuple[float, float]]
 ):
 self.model = model
 self.feature_names = feature_names
 self.feature_ranges = feature_ranges
 
 def generate(
 self,
 transaction: torch.Tensor,
 target_class: int,
 max_changes: int = 3,
 max_iterations: int = 100
 ) -> Optional[Dict[str, Any]]:
 """
 Find minimal changes to flip prediction
 
 Args:
 transaction: Original transaction
 target_class: Desired class (0=legit, 1=fraud)
 max_changes: Maximum features to modify
 max_iterations: Search iterations
 
 Returns:
 {
 'changes': Dict[feature_name] = new_value,
 'confidence': float
 }
 """
 self.model.eval()
 
 # Start from original
 counterfactual = transaction.clone()
 
 # Get baseline prediction
 with torch.no_grad():
 baseline_pred = self.model.predict(transaction).item()
 
 if baseline_pred == target_class:
 return None # Already in target class
 
 # Iteratively modify features with progress tracking
 pbar = tqdm(range(max_iterations), 
 desc=f" Buscando contrafactual (alvo={target_class})", 
 unit="iter",
 leave=False)
 
 for iteration in pbar:
 # Get current prediction
 with torch.no_grad():
 current_pred = self.model.predict(counterfactual).item()
 current_proba = self.model.predict_proba(counterfactual)[0, target_class].item()
 
 # Update progress bar with current probability
 pbar.set_postfix({'prob': f'{current_proba:.3f}'})
 
 # Check if flipped
 if current_pred == target_class and current_proba > 0.7:
 # Success! Return changes
 changes = {}
 for i, feature_name in enumerate(self.feature_names):
 if abs(transaction[0, i].item() - counterfactual[0, i].item()) > 0.01:
 changes[feature_name] = counterfactual[0, i].item()
 
 if len(changes) >= max_changes:
 break
 
 pbar.close()
 return {
 'changes': changes,
 'confidence': current_proba
 }
 
 # Gradient-based search
 counterfactual.requires_grad = True
 output = self.model.predict_proba(counterfactual)
 loss = -output[0, target_class] # Maximize target class probability
 loss.backward()
 
 # Update
 with torch.no_grad():
 counterfactual -= 0.1 * counterfactual.grad.sign()
 counterfactual.grad.zero_()
 
 # Clip to valid ranges
 for i, feature_name in enumerate(self.feature_names):
 if feature_name in self.feature_ranges:
 min_val, max_val = self.feature_ranges[feature_name]
 counterfactual[0, i] = torch.clamp(
 counterfactual[0, i],
 min=min_val,
 max=max_val
 )
 
 return None # Failed to find counterfactual

class ExplainabilityEngine:
 """
 Complete explainability system
 
 Combines:
 - SHAP
 - Ablation
 - Spike patterns
 - Counterfactuals
 """
 
 def __init__(
 self,
 model: nn.Module,
 background_data: torch.Tensor,
 feature_names: List[str],
 feature_ranges: Optional[Dict[str, Tuple[float, float]]] = None
 ):
 self.model = model
 self.feature_names = feature_names
 
 # Initialize explainers
 self.shap_explainer = SHAPExplainer(model, background_data, feature_names)
 self.ablation_explainer = AblationExplainer(model, feature_names)
 self.spike_analyzer = SpikePatternAnalyzer(model)
 
 if feature_ranges:
 self.counterfactual_generator = CounterfactualGenerator(
 model, feature_names, feature_ranges
 )
 else:
 self.counterfactual_generator = None
 
 def explain_prediction(
 self,
 transaction: torch.Tensor,
 transaction_id: str = "unknown"
 ) -> Explanation:
 """
 Generate complete explanation for a prediction
 
 This is the main API for LGPD/GDPR compliance
 """
 logger.info(f"Generating explanation for transaction {transaction_id}")
 
 # Prediction
 prediction = self.model.predict(transaction).item()
 proba = self.model.predict_proba(transaction)
 confidence = proba[0, prediction].item()
 
 # SHAP values
 shap_values = self.shap_explainer.explain(transaction)
 
 # Feature importance (ablation)
 feature_importance = self.ablation_explainer.explain(transaction)
 
 # Spike pattern
 spike_pattern = self.spike_analyzer.analyze(transaction)
 
 # Counterfactual
 counterfactual = None
 if self.counterfactual_generator:
 target_class = 1 - prediction # Flip class
 counterfactual = self.counterfactual_generator.generate(
 transaction, target_class
 )
 
 # Create explanation object
 explanation = Explanation(
 transaction_id=transaction_id,
 prediction=prediction,
 confidence=confidence,
 feature_importance=feature_importance,
 shap_values=shap_values,
 spike_pattern=spike_pattern,
 counterfactual=counterfactual
 )
 
 return explanation
 
 def generate_report(
 self,
 explanation: Explanation,
 save_path: Optional[Path] = None
 ) -> str:
 """
 Generate PDF/HTML report for compliance
 """
 report = []
 
 report.append("=" * 60)
 report.append("FRAUD DETECTION EXPLANATION REPORT")
 report.append("=" * 60)
 report.append("")
 report.append(f"Transaction ID: {explanation.transaction_id}")
 report.append(f"Prediction: {'FRAUD' if explanation.prediction == 1 else 'LEGIT'}")
 report.append(f"Confidence: {explanation.confidence*100:.2f}%")
 report.append("")
 
 report.append("TOP 5 MOST IMPORTANT FEATURES:")
 report.append("-" * 60)
 for i, (feature, importance) in enumerate(list(explanation.feature_importance.items())[:5], 1):
 report.append(f" {i}. {feature:30s} {importance:.4f}")
 report.append("")
 
 report.append("SPIKE PATTERN ANALYSIS:")
 report.append("-" * 60)
 report.append(f" Total spikes: {explanation.spike_pattern['total_spikes']}")
 report.append(f" Spike rate: {explanation.spike_pattern['spike_rate']:.4f}")
 report.append("")
 
 if explanation.counterfactual:
 report.append("COUNTERFACTUAL EXPLANATION:")
 report.append("-" * 60)
 report.append(" To change the classification, modify:")
 for feature, value in explanation.counterfactual['changes'].items():
 report.append(f" {feature}: {value:.4f}")
 report.append("")
 
 report.append("=" * 60)
 report.append("This explanation complies with LGPD Art. 20 (right to explanation)")
 report.append("=" * 60)
 
 report_text = "\n".join(report)
 
 if save_path:
 save_path.write_text(report_text)
 
 return report_text

if __name__ == "__main__":
 # Demo
 print("Explainability Module for SNN Fraud Detection")
 print("-" * 60)
 
 # Mock data
 from src.models_snn_pytorch import FraudSNNPyTorch
 
 model = FraudSNNPyTorch(input_size=64, hidden_sizes=[32, 16], output_size=2)
 background_data = torch.randn(100, 64)
 feature_names = [f"feature_{i}" for i in range(64)]
 
 # Create explainer
 explainer = ExplainabilityEngine(
 model=model,
 background_data=background_data,
 feature_names=feature_names
 )
 
 # Explain a transaction
 transaction = torch.randn(1, 64)
 explanation = explainer.explain_prediction(transaction, "TXN_12345")
 
 # Generate report
 report = explainer.generate_report(explanation)
 print(report)
