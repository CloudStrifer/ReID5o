"""
Reliability-Adaptive Fusion Module

This module implements intelligent fusion that dynamically determines which modalities
are more trustworthy and adjusts fusion weights accordingly. It upgrades from 
"passive fusion" to "active decision-making."

Key Components:
1. ModalityReliabilityEstimator: Estimates confidence/reliability score for each modality
2. ReliabilityAdaptiveFusion: Performs weighted fusion based on reliability scores
3. SparsityRegularization: Encourages relying on fewer high-quality modalities
4. UncertaintyAwareLoss: Penalizes high weights on low-quality modalities

The goal: the model learns to automatically choose different strategies in different 
scenarios (e.g., rely more on IR at night, rely more on text when sufficient).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Dict, Tuple, Optional


class LayerNorm(nn.LayerNorm):
    """LayerNorm that handles fp16 properly"""
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        weight_dtype = self.weight.dtype
        x = x.to(weight_dtype)
        ret = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return ret.type(orig_type) if orig_type != weight_dtype else ret


class QuickGELU(nn.Module):
    """Fast GELU activation"""
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ModalityQualityIndicators(nn.Module):
    """
    Computes quality indicators from modality features.
    
    These indicators reflect the quality/reliability of each modality:
    - Feature variance: Higher variance may indicate more information
    - Attention entropy: How concentrated the attention is
    - Feature norm: Magnitude of the features
    - Prediction confidence: How confident the feature is for identity
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Learnable projections for quality assessment
        self.variance_proj = nn.Linear(embed_dim, embed_dim)
        self.confidence_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            QuickGELU(),
            nn.Linear(embed_dim // 2, 1)
        )
        
    def compute_feature_variance(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute normalized variance of features.
        Higher variance often indicates richer information.
        
        Args:
            features: (batch_size, embed_dim)
            
        Returns:
            Variance score (batch_size, 1)
        """
        # Project features
        projected = self.variance_proj(features)
        # Compute variance across feature dimension
        variance = torch.var(projected, dim=-1, keepdim=True)
        # Normalize to [0, 1] range using sigmoid
        normalized_var = torch.sigmoid(variance)
        return normalized_var
    
    def compute_feature_norm(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute L2 norm of features (normalized).
        
        Args:
            features: (batch_size, embed_dim)
            
        Returns:
            Norm score (batch_size, 1)
        """
        norm = torch.norm(features, p=2, dim=-1, keepdim=True)
        # Normalize by expected norm (sqrt of embed_dim for unit variance)
        expected_norm = math.sqrt(self.embed_dim)
        normalized_norm = norm / expected_norm
        # Clamp and apply sigmoid
        return torch.sigmoid(normalized_norm - 1.0)  # Center around 1.0
    
    def compute_prediction_confidence(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute prediction confidence from features.
        
        Args:
            features: (batch_size, embed_dim)
            
        Returns:
            Confidence score (batch_size, 1)
        """
        confidence = self.confidence_proj(features)
        return torch.sigmoid(confidence)
    
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute all quality indicators.
        
        Args:
            features: (batch_size, embed_dim)
            
        Returns:
            Dict of quality indicators
        """
        return {
            'variance': self.compute_feature_variance(features),
            'norm': self.compute_feature_norm(features),
            'confidence': self.compute_prediction_confidence(features)
        }


class ModalityReliabilityEstimator(nn.Module):
    """
    Estimates the reliability/confidence score for each modality.
    
    The reliability score indicates how trustworthy a modality's features are,
    considering factors like:
    - Feature quality indicators (variance, norm, confidence)
    - Whether the feature is real or generated (completed)
    - Learned modality-specific priors
    """
    def __init__(self, 
                 embed_dim: int,
                 num_modalities: int = 5,
                 hidden_dim: int = 256,
                 use_quality_indicators: bool = True):
        """
        Args:
            embed_dim: Feature embedding dimension (512 for CLIP)
            num_modalities: Number of modalities (RGB, NIR, CP, SK, TEXT = 5)
            hidden_dim: Hidden dimension for reliability network
            use_quality_indicators: Whether to use computed quality indicators
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_modalities = num_modalities
        self.use_quality_indicators = use_quality_indicators
        
        # Modality names
        self.modality_names = ['RGB', 'NIR', 'CP', 'SK', 'TEXT']
        
        # Quality indicator computation
        if use_quality_indicators:
            self.quality_indicators = ModalityQualityIndicators(embed_dim)
            indicator_dim = 3  # variance, norm, confidence
        else:
            self.quality_indicators = None
            indicator_dim = 0
        
        # Input dimension: features + quality indicators + is_generated flag
        input_dim = embed_dim + indicator_dim + 1  # +1 for is_generated flag
        
        # Reliability estimation network (per-modality)
        self.reliability_networks = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                LayerNorm(hidden_dim),
                QuickGELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim // 2),
                QuickGELU(),
                nn.Linear(hidden_dim // 2, 1)
            )
            for name in self.modality_names
        })
        
        # Learnable modality-specific bias (prior reliability)
        self.modality_bias = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(1))
            for name in self.modality_names
        })
        
        # Temperature for softmax
        self.temperature = nn.Parameter(torch.ones(1))
        
        self._init_weights()
        
    def _init_weights(self):
        for name, network in self.reliability_networks.items():
            for m in network.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
    
    def estimate_single_reliability(self,
                                     features: torch.Tensor,
                                     modality_name: str,
                                     is_generated: bool = False) -> torch.Tensor:
        """
        Estimate reliability for a single modality.
        
        Args:
            features: Feature tensor (batch_size, embed_dim)
            modality_name: Name of the modality
            is_generated: Whether features are generated (completed)
            
        Returns:
            Reliability score (batch_size, 1)
        """
        batch_size = features.shape[0]
        device = features.device
        
        # Ensure features are in the right dtype
        module_dtype = next(self.parameters()).dtype
        features = features.to(dtype=module_dtype)
        
        # Compute quality indicators
        if self.use_quality_indicators:
            indicators = self.quality_indicators(features)
            indicator_tensor = torch.cat([
                indicators['variance'],
                indicators['norm'],
                indicators['confidence']
            ], dim=-1)
        else:
            indicator_tensor = torch.zeros(batch_size, 0, device=device, dtype=module_dtype)
        
        # Is-generated flag
        generated_flag = torch.full((batch_size, 1), float(is_generated), 
                                     device=device, dtype=module_dtype)
        
        # Concatenate all inputs
        network_input = torch.cat([features, indicator_tensor, generated_flag], dim=-1)
        
        # Compute reliability score
        reliability = self.reliability_networks[modality_name](network_input)
        
        # Add modality-specific bias
        reliability = reliability + self.modality_bias[modality_name]
        
        return reliability
    
    def forward(self,
                modality_features: Dict[str, torch.Tensor],
                is_generated: Dict[str, bool] = None) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Estimate reliability scores for all modalities and compute fusion weights.
        
        Args:
            modality_features: Dict mapping modality name to features (batch_size, embed_dim)
            is_generated: Dict mapping modality name to whether it's generated
            
        Returns:
            Tuple of:
            - Dict of reliability scores per modality
            - Fusion weights tensor (batch_size, num_available_modalities)
        """
        if is_generated is None:
            is_generated = {name: False for name in self.modality_names}
        
        reliability_scores = {}
        score_list = []
        modality_order = []
        
        for name in self.modality_names:
            if name in modality_features:
                features = modality_features[name]
                generated = is_generated.get(name, False)
                
                score = self.estimate_single_reliability(features, name, generated)
                reliability_scores[name] = score
                score_list.append(score)
                modality_order.append(name)
        
        if len(score_list) == 0:
            raise ValueError("At least one modality must be provided")
        
        # Stack scores and compute softmax weights
        stacked_scores = torch.cat(score_list, dim=-1)  # (batch_size, num_modalities)
        
        # Temperature-scaled softmax
        temperature = torch.clamp(self.temperature, min=0.1)
        fusion_weights = F.softmax(stacked_scores / temperature, dim=-1)
        
        return reliability_scores, fusion_weights, modality_order


class ReliabilityAdaptiveFusion(nn.Module):
    """
    Performs adaptive fusion based on reliability scores.
    
    Instead of simple averaging or concatenation, this module:
    1. Estimates reliability of each modality (real or generated)
    2. Computes adaptive weights via softmax
    3. Performs weighted fusion
    4. Optionally applies transformer for further refinement
    """
    def __init__(self,
                 embed_dim: int,
                 num_modalities: int = 5,
                 hidden_dim: int = 256,
                 num_heads: int = 8,
                 num_fusion_layers: int = 2,
                 use_quality_indicators: bool = True,
                 use_transformer_refinement: bool = True):
        """
        Args:
            embed_dim: Feature embedding dimension
            num_modalities: Number of modalities
            hidden_dim: Hidden dimension for reliability estimator
            num_heads: Number of attention heads for transformer refinement
            num_fusion_layers: Number of transformer layers for refinement
            use_quality_indicators: Whether to use quality indicators
            use_transformer_refinement: Whether to use transformer for post-fusion refinement
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_modalities = num_modalities
        self.modality_names = ['RGB', 'NIR', 'CP', 'SK', 'TEXT']
        
        # Reliability estimator
        self.reliability_estimator = ModalityReliabilityEstimator(
            embed_dim=embed_dim,
            num_modalities=num_modalities,
            hidden_dim=hidden_dim,
            use_quality_indicators=use_quality_indicators
        )
        
        # Optional transformer for post-fusion refinement
        self.use_transformer_refinement = use_transformer_refinement
        if use_transformer_refinement:
            self.fusion_transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=embed_dim * 4,
                    dropout=0.1,
                    activation='gelu',
                    batch_first=True
                ),
                num_layers=num_fusion_layers
            )
            self.fusion_ln = LayerNorm(embed_dim)
        
        # Output projection
        self.output_proj = nn.Sequential(
            LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            QuickGELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Learnable fusion query token
        self.fusion_query = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.fusion_query, std=0.02)
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.output_proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def weighted_fusion(self,
                        modality_features: Dict[str, torch.Tensor],
                        fusion_weights: torch.Tensor,
                        modality_order: List[str]) -> torch.Tensor:
        """
        Perform weighted fusion of modality features.
        
        Args:
            modality_features: Dict of features per modality
            fusion_weights: Weights tensor (batch_size, num_modalities)
            modality_order: Order of modalities in weights tensor
            
        Returns:
            Fused features (batch_size, embed_dim)
        """
        # Stack features in the same order as weights
        feature_list = [modality_features[name] for name in modality_order]
        stacked_features = torch.stack(feature_list, dim=1)  # (batch_size, num_modalities, embed_dim)
        
        # Ensure same dtype
        module_dtype = next(self.parameters()).dtype
        stacked_features = stacked_features.to(dtype=module_dtype)
        fusion_weights = fusion_weights.to(dtype=module_dtype)
        
        # Weighted sum
        weights_expanded = fusion_weights.unsqueeze(-1)  # (batch_size, num_modalities, 1)
        fused = (stacked_features * weights_expanded).sum(dim=1)  # (batch_size, embed_dim)
        
        return fused
    
    def forward(self,
                modality_features: Dict[str, torch.Tensor],
                is_generated: Dict[str, bool] = None,
                return_weights: bool = False) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Perform reliability-adaptive fusion.
        
        Args:
            modality_features: Dict mapping modality name to features
            is_generated: Dict indicating which modalities are generated
            return_weights: Whether to return fusion weights and reliability scores
            
        Returns:
            Tuple of:
            - Fused features (batch_size, embed_dim)
            - Optional dict with weights and scores (if return_weights=True)
        """
        # Estimate reliability and get fusion weights
        reliability_scores, fusion_weights, modality_order = self.reliability_estimator(
            modality_features, is_generated
        )
        
        # Perform weighted fusion
        fused = self.weighted_fusion(modality_features, fusion_weights, modality_order)
        
        # Optional transformer refinement
        if self.use_transformer_refinement:
            batch_size = fused.shape[0]
            
            # Prepare input: fusion query + fused features
            query = self.fusion_query.expand(batch_size, -1, -1)
            
            # Stack all features for context
            feature_list = [modality_features[name].unsqueeze(1) for name in modality_order]
            context = torch.cat(feature_list, dim=1)  # (batch_size, num_modalities, embed_dim)
            
            # Ensure same dtype
            module_dtype = next(self.parameters()).dtype
            query = query.to(dtype=module_dtype)
            context = context.to(dtype=module_dtype)
            
            # Concatenate query with context
            transformer_input = torch.cat([query, context], dim=1)
            
            # Apply transformer
            refined = self.fusion_transformer(transformer_input)
            
            # Take the query position output
            fused = self.fusion_ln(refined[:, 0, :])
        
        # Output projection
        output = self.output_proj(fused)
        
        # Convert to float32 for downstream tasks
        output = output.float()
        
        if return_weights:
            info = {
                'reliability_scores': {k: v.float() for k, v in reliability_scores.items()},
                'fusion_weights': fusion_weights.float(),
                'modality_order': modality_order
            }
            return output, info
        
        return output, None


class SparsityRegularization(nn.Module):
    """
    Sparsity regularization for fusion weights.
    
    Encourages the model to rely on fewer high-quality modalities
    rather than using all modalities equally.
    """
    def __init__(self, 
                 target_sparsity: float = 0.3,
                 regularization_type: str = 'entropy'):
        """
        Args:
            target_sparsity: Target sparsity level (0 = all equal, 1 = single modality)
            regularization_type: Type of regularization ('entropy', 'l1', 'gini')
        """
        super().__init__()
        self.target_sparsity = target_sparsity
        self.regularization_type = regularization_type
        
    def compute_entropy(self, weights: torch.Tensor) -> torch.Tensor:
        """Compute entropy of weight distribution."""
        # Add small epsilon to avoid log(0)
        eps = 1e-8
        entropy = -torch.sum(weights * torch.log(weights + eps), dim=-1)
        return entropy.mean()
    
    def compute_gini(self, weights: torch.Tensor) -> torch.Tensor:
        """Compute Gini coefficient (measure of inequality/sparsity)."""
        sorted_weights, _ = torch.sort(weights, dim=-1)
        n = weights.shape[-1]
        indices = torch.arange(1, n + 1, device=weights.device, dtype=weights.dtype)
        gini = (2 * torch.sum(indices * sorted_weights, dim=-1) / (n * torch.sum(sorted_weights, dim=-1))) - (n + 1) / n
        return -gini.mean()  # Negative because we want to maximize sparsity
    
    def forward(self, fusion_weights: torch.Tensor) -> torch.Tensor:
        """
        Compute sparsity regularization loss.
        
        Args:
            fusion_weights: Fusion weights (batch_size, num_modalities)
            
        Returns:
            Sparsity regularization loss
        """
        if self.regularization_type == 'entropy':
            # Minimize entropy to encourage sparsity
            entropy = self.compute_entropy(fusion_weights)
            # Target entropy based on sparsity
            num_modalities = fusion_weights.shape[-1]
            max_entropy = math.log(num_modalities)
            target_entropy = max_entropy * (1 - self.target_sparsity)
            loss = F.relu(entropy - target_entropy)
            
        elif self.regularization_type == 'l1':
            # L1 regularization on weights (encourages sparsity)
            loss = torch.mean(torch.sum(torch.abs(fusion_weights), dim=-1))
            
        elif self.regularization_type == 'gini':
            # Gini coefficient (higher = more sparse)
            gini_loss = self.compute_gini(fusion_weights)
            target_gini = self.target_sparsity
            loss = F.relu(target_gini - (-gini_loss))
            
        else:
            raise ValueError(f"Unknown regularization type: {self.regularization_type}")
        
        return loss


class UncertaintyAwareLoss(nn.Module):
    """
    Uncertainty-aware loss that penalizes high weight allocation 
    to low-quality/uncertain modalities.
    
    This encourages the model to:
    - Assign low weights to generated (completed) features
    - Assign low weights to features with high uncertainty
    """
    def __init__(self, 
                 generated_penalty: float = 0.5,
                 uncertainty_threshold: float = 0.5):
        """
        Args:
            generated_penalty: Penalty multiplier for generated modalities
            uncertainty_threshold: Threshold below which features are considered uncertain
        """
        super().__init__()
        self.generated_penalty = generated_penalty
        self.uncertainty_threshold = uncertainty_threshold
        
    def forward(self,
                fusion_weights: torch.Tensor,
                reliability_scores: Dict[str, torch.Tensor],
                is_generated: Dict[str, bool],
                modality_order: List[str]) -> torch.Tensor:
        """
        Compute uncertainty-aware loss.
        
        Args:
            fusion_weights: Fusion weights (batch_size, num_modalities)
            reliability_scores: Dict of reliability scores per modality
            is_generated: Dict indicating which modalities are generated
            modality_order: Order of modalities in weights tensor
            
        Returns:
            Uncertainty-aware loss
        """
        device = fusion_weights.device
        batch_size = fusion_weights.shape[0]
        
        total_loss = torch.tensor(0.0, device=device)
        
        for i, name in enumerate(modality_order):
            weight = fusion_weights[:, i]  # (batch_size,)
            
            # Penalty for generated modalities
            if is_generated.get(name, False):
                # Penalize high weights on generated features
                generated_loss = weight.mean() * self.generated_penalty
                total_loss = total_loss + generated_loss
            
            # Penalty based on low reliability
            if name in reliability_scores:
                reliability = reliability_scores[name].squeeze(-1)  # (batch_size,)
                # Convert to same dtype
                reliability = reliability.to(dtype=weight.dtype)
                
                # Penalize high weight when reliability is low
                # Loss = weight * (1 - reliability) when reliability < threshold
                uncertainty = F.relu(self.uncertainty_threshold - torch.sigmoid(reliability))
                uncertainty_loss = (weight * uncertainty).mean()
                total_loss = total_loss + uncertainty_loss
        
        return total_loss


class ReliabilityAdaptiveFusionTrainer(nn.Module):
    """
    Training helper for reliability-adaptive fusion.
    
    Combines:
    - Sparsity regularization
    - Uncertainty-aware loss
    - Feature reconstruction guidance
    """
    def __init__(self,
                 fusion_module: ReliabilityAdaptiveFusion,
                 sparsity_weight: float = 0.1,
                 uncertainty_weight: float = 0.2,
                 sparsity_target: float = 0.3,
                 sparsity_type: str = 'entropy'):
        """
        Args:
            fusion_module: The ReliabilityAdaptiveFusion module
            sparsity_weight: Weight for sparsity regularization loss
            uncertainty_weight: Weight for uncertainty-aware loss
            sparsity_target: Target sparsity level
            sparsity_type: Type of sparsity regularization
        """
        super().__init__()
        self.fusion_module = fusion_module
        self.sparsity_weight = sparsity_weight
        self.uncertainty_weight = uncertainty_weight
        
        self.sparsity_reg = SparsityRegularization(
            target_sparsity=sparsity_target,
            regularization_type=sparsity_type
        )
        self.uncertainty_loss = UncertaintyAwareLoss()
        
    def forward(self,
                modality_features: Dict[str, torch.Tensor],
                is_generated: Dict[str, bool] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Perform fusion and compute training losses.
        
        Args:
            modality_features: Dict of features per modality
            is_generated: Dict indicating which modalities are generated
            
        Returns:
            Tuple of:
            - Fused features
            - Dict of losses
        """
        if is_generated is None:
            is_generated = {name: False for name in self.fusion_module.modality_names}
        
        # Perform fusion with weight tracking
        fused, info = self.fusion_module(
            modality_features, is_generated, return_weights=True
        )
        
        losses = {}
        
        # Sparsity regularization
        sparsity_loss = self.sparsity_reg(info['fusion_weights'])
        losses['fusion_sparsity_loss'] = sparsity_loss * self.sparsity_weight
        
        # Uncertainty-aware loss
        uncertainty_loss = self.uncertainty_loss(
            info['fusion_weights'],
            info['reliability_scores'],
            is_generated,
            info['modality_order']
        )
        losses['fusion_uncertainty_loss'] = uncertainty_loss * self.uncertainty_weight
        
        return fused, losses, info


class AdaptiveFusionInferenceHelper:
    """
    Helper for using adaptive fusion during inference.
    
    Provides utilities for:
    - Fusing features with automatic reliability estimation
    - Analyzing which modalities are being relied upon
    - Debugging and visualization
    """
    def __init__(self, fusion_module: ReliabilityAdaptiveFusion):
        self.fusion_module = fusion_module
        
    def fuse_with_analysis(self,
                           modality_features: Dict[str, torch.Tensor],
                           is_generated: Dict[str, bool] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Perform fusion with detailed analysis.
        
        Returns fused features along with:
        - Reliability scores per modality
        - Fusion weights
        - Dominant modality
        """
        fused, info = self.fusion_module(modality_features, is_generated, return_weights=True)
        
        # Find dominant modality
        weights = info['fusion_weights']
        modality_order = info['modality_order']
        
        # Average weights across batch
        avg_weights = weights.mean(dim=0)
        dominant_idx = torch.argmax(avg_weights).item()
        dominant_modality = modality_order[dominant_idx]
        
        analysis = {
            'fused_features': fused,
            'reliability_scores': info['reliability_scores'],
            'fusion_weights': info['fusion_weights'],
            'modality_order': modality_order,
            'dominant_modality': dominant_modality,
            'weight_distribution': {
                name: avg_weights[i].item() 
                for i, name in enumerate(modality_order)
            }
        }
        
        return fused, analysis
    
    def get_modality_importance(self,
                                 modality_features: Dict[str, torch.Tensor],
                                 is_generated: Dict[str, bool] = None) -> Dict[str, float]:
        """
        Get importance ranking of modalities.
        """
        _, info = self.fusion_module(modality_features, is_generated, return_weights=True)
        
        weights = info['fusion_weights'].mean(dim=0)
        modality_order = info['modality_order']
        
        importance = {
            name: weights[i].item()
            for i, name in enumerate(modality_order)
        }
        
        # Sort by importance
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        
        return importance
