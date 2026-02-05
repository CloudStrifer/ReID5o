"""
Cross-modal Feature Completion Module

This module implements feature-level completion for missing modalities in multi-modal 
person re-identification. Instead of generating actual images (too heavy and unstable),
we generate discriminative feature embeddings directly.

Key Components:
1. ModalityFeatureGenerator: Conditional generator that predicts missing modality features
2. CrossModalCompletionModule: Main module that manages all modality generators
3. CycleConsistencyLoss: Ensures semantic consistency through cycle transformations

The workflow:
- During training: Randomly mask modalities and use real features as supervision
- During inference: Complete missing modality features using available modalities

This enables "information recovery" rather than just "survival with missing modalities".
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Dict, Tuple, Optional
from collections import OrderedDict


class QuickGELU(nn.Module):
    """Fast GELU activation (same as CLIP)"""
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class LayerNorm(nn.LayerNorm):
    """LayerNorm that handles fp16 properly"""
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        # Use the same dtype as the weight for layer norm computation
        weight_dtype = self.weight.dtype
        x = x.to(weight_dtype)
        ret = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return ret.type(orig_type) if orig_type != weight_dtype else ret


class CrossAttentionBlock(nn.Module):
    """
    Cross-attention block for conditional feature generation.
    Query: target modality representation
    Key/Value: source modality features
    """
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ln_cross = LayerNorm(embed_dim)
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ln_self = LayerNorm(embed_dim)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            QuickGELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        self.ln_ffn = LayerNorm(embed_dim)
        
    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: Target modality query (batch_size, 1, embed_dim) or (batch_size, embed_dim)
            context: Source modality features (batch_size, seq_len, embed_dim)
            
        Returns:
            Updated query representation
        """
        # Ensure query is 3D
        if query.dim() == 2:
            query = query.unsqueeze(1)
        
        # Cross-attention: query attends to context
        residual = query
        query = self.ln_cross(query)
        context_normed = self.ln_cross(context)
        query = residual + self.cross_attn(query, context_normed, context_normed, need_weights=False)[0]
        
        # Self-attention on query
        residual = query
        query = self.ln_self(query)
        query = residual + self.self_attn(query, query, query, need_weights=False)[0]
        
        # FFN
        residual = query
        query = residual + self.ffn(self.ln_ffn(query))
        
        return query


class ModalityFeatureGenerator(nn.Module):
    """
    Conditional generator for a specific target modality.
    
    Given features from available modalities (context), generates the feature
    embedding for the target modality.
    """
    def __init__(self, 
                 embed_dim: int,
                 num_heads: int = 8,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        """
        Args:
            embed_dim: Dimension of feature embeddings (512 for CLIP)
            num_heads: Number of attention heads
            num_layers: Number of cross-attention layers
            dropout: Dropout rate
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        
        # Learnable query token for the target modality
        self.query_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.query_token, std=0.02)
        
        # Cross-attention layers
        self.layers = nn.ModuleList([
            CrossAttentionBlock(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            QuickGELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Initialize output projection
        self._init_weights()
        
    def _init_weights(self):
        for m in self.output_proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        Generate target modality feature from context.
        
        Args:
            context: Fused features from available modalities (batch_size, seq_len, embed_dim)
                     or CLS token (batch_size, embed_dim)
                     
        Returns:
            Generated feature for target modality (batch_size, embed_dim)
        """
        batch_size = context.shape[0]
        
        # Ensure context is 3D
        if context.dim() == 2:
            context = context.unsqueeze(1)
        
        # Expand query token to batch size
        query = self.query_token.expand(batch_size, -1, -1)
        
        # Apply cross-attention layers
        for layer in self.layers:
            query = layer(query, context)
        
        # Extract CLS-like token and project
        output = query.squeeze(1)  # (batch_size, embed_dim)
        output = self.output_proj(output)
        
        return output


class CrossModalCompletionModule(nn.Module):
    """
    Main Cross-modal Feature Completion Module.
    
    Manages generators for all modalities and handles:
    1. Feature generation for missing modalities
    2. Cycle consistency enforcement
    3. Reconstruction loss computation
    """
    def __init__(self,
                 embed_dim: int,
                 num_modalities: int = 5,
                 num_heads: int = 8,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        """
        Args:
            embed_dim: Feature embedding dimension (512 for CLIP)
            num_modalities: Number of modalities (RGB, NIR, CP, SK, TEXT = 5)
            num_heads: Number of attention heads in generators
            num_layers: Number of layers in each generator
            dropout: Dropout rate
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_modalities = num_modalities
        
        # Modality names for reference
        self.modality_names = ['RGB', 'NIR', 'CP', 'SK', 'TEXT']
        
        # Create a generator for each modality
        # Generator i produces features for modality i given other modalities
        self.generators = nn.ModuleDict({
            name: ModalityFeatureGenerator(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout=dropout
            )
            for name in self.modality_names
        })
        
        # Context fusion module: combines available modalities into a context
        self.context_fusion = nn.Sequential(
            LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            QuickGELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Modality-specific context projections
        self.modality_context_proj = nn.ModuleDict({
            name: nn.Linear(embed_dim, embed_dim)
            for name in self.modality_names
        })
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.context_fusion.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for proj in self.modality_context_proj.values():
            nn.init.xavier_uniform_(proj.weight)
            nn.init.zeros_(proj.bias)
    
    def fuse_context(self, 
                     modality_features: Dict[str, torch.Tensor],
                     modality_mask: List[bool]) -> torch.Tensor:
        """
        Fuse available modality features into a context representation.
        
        Args:
            modality_features: Dict mapping modality name to feature tensor (batch_size, embed_dim)
            modality_mask: List of booleans indicating which modalities are present
            
        Returns:
            Fused context tensor (batch_size, num_available, embed_dim)
        """
        available_features = []
        
        # Get the dtype of the module weights for proper conversion
        module_dtype = next(self.parameters()).dtype
        
        for i, (name, present) in enumerate(zip(self.modality_names, modality_mask)):
            if present and name in modality_features:
                feat = modality_features[name]
                # Convert to module dtype before projection
                feat = feat.to(dtype=module_dtype)
                # Project to context space
                projected = self.modality_context_proj[name](feat)
                available_features.append(projected)
        
        if len(available_features) == 0:
            raise ValueError("At least one modality must be present for context")
        
        # Stack available features: (batch_size, num_available, embed_dim)
        context = torch.stack(available_features, dim=1)
        
        return context
    
    def generate_missing_features(self,
                                   modality_features: Dict[str, torch.Tensor],
                                   modality_mask: List[bool]) -> Dict[str, torch.Tensor]:
        """
        Generate features for all missing modalities.
        
        Args:
            modality_features: Dict of available modality features
            modality_mask: List indicating which modalities are present
            
        Returns:
            Dict of generated features for missing modalities (in float32 for loss computation)
        """
        # Fuse available features into context
        context = self.fuse_context(modality_features, modality_mask)
        
        # Apply context fusion
        context = self.context_fusion(context)
        
        # Generate features for missing modalities
        generated_features = {}
        
        for i, (name, present) in enumerate(zip(self.modality_names, modality_mask)):
            if not present:
                # Generate feature for this missing modality
                generated = self.generators[name](context)
                # Convert to float32 for loss computation
                generated_features[name] = generated.float()
        
        return generated_features
    
    def complete_features(self,
                          modality_features: Dict[str, torch.Tensor],
                          modality_mask: List[bool]) -> Dict[str, torch.Tensor]:
        """
        Complete the feature set by adding generated features for missing modalities.
        
        Args:
            modality_features: Dict of available modality features
            modality_mask: List indicating which modalities are present
            
        Returns:
            Complete dict with both real and generated features
        """
        # Start with real features
        complete_features = dict(modality_features)
        
        # Generate missing features
        generated = self.generate_missing_features(modality_features, modality_mask)
        
        # Add generated features (with a flag or just merge)
        complete_features.update(generated)
        
        return complete_features


class FeatureReconstructionLoss(nn.Module):
    """
    Reconstruction loss for training the completion generators.
    
    During training, we mask real features and try to reconstruct them
    using the generator. The loss measures how close the generated
    features are to the real features.
    """
    def __init__(self, loss_type: str = 'cosine', margin: float = 0.0):
        """
        Args:
            loss_type: Type of loss ('l2', 'l1', 'cosine', 'combined')
            margin: Margin for contrastive-style losses
        """
        super().__init__()
        self.loss_type = loss_type
        self.margin = margin
        
    def forward(self, 
                generated: torch.Tensor, 
                target: torch.Tensor,
                normalize: bool = True) -> torch.Tensor:
        """
        Compute reconstruction loss.
        
        Args:
            generated: Generated features (batch_size, embed_dim)
            target: Real features (batch_size, embed_dim)
            normalize: Whether to normalize features before computing loss
            
        Returns:
            Reconstruction loss scalar
        """
        # Ensure both tensors are in float32 for stable loss computation
        generated = generated.float()
        target = target.float()
        
        if normalize:
            generated = F.normalize(generated, p=2, dim=-1)
            target = F.normalize(target, p=2, dim=-1)
        
        if self.loss_type == 'l2':
            loss = F.mse_loss(generated, target)
        elif self.loss_type == 'l1':
            loss = F.l1_loss(generated, target)
        elif self.loss_type == 'cosine':
            # Cosine similarity loss: maximize similarity
            cos_sim = F.cosine_similarity(generated, target, dim=-1)
            loss = (1 - cos_sim).mean()
        elif self.loss_type == 'combined':
            # Combine L2 and cosine
            l2_loss = F.mse_loss(generated, target)
            cos_loss = (1 - F.cosine_similarity(generated, target, dim=-1)).mean()
            loss = l2_loss + cos_loss
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return loss


class CycleConsistencyLoss(nn.Module):
    """
    Cycle consistency loss for stable cross-modal generation.
    
    The idea: if we go text → rgb → text, the result should still be 
    close to the original text embedding. This prevents semantic drift
    in generated features.
    
    Forward cycle: A → B → A (should match original A)
    """
    def __init__(self, loss_type: str = 'cosine'):
        super().__init__()
        self.loss_type = loss_type
        self.recon_loss = FeatureReconstructionLoss(loss_type=loss_type)
        
    def forward(self,
                original: torch.Tensor,
                reconstructed: torch.Tensor) -> torch.Tensor:
        """
        Compute cycle consistency loss.
        
        Args:
            original: Original feature (batch_size, embed_dim)
            reconstructed: Feature after cycle transformation (batch_size, embed_dim)
            
        Returns:
            Cycle consistency loss
        """
        return self.recon_loss(reconstructed, original, normalize=True)


class CrossModalCompletionTrainer(nn.Module):
    """
    Training helper for cross-modal completion.
    
    Handles:
    1. Random masking of modalities during training
    2. Reconstruction loss computation
    3. Cycle consistency loss computation
    4. Combining real and generated features for downstream tasks
    """
    def __init__(self,
                 completion_module: CrossModalCompletionModule,
                 reconstruction_loss_weight: float = 1.0,
                 cycle_loss_weight: float = 0.5,
                 loss_type: str = 'cosine'):
        """
        Args:
            completion_module: The CrossModalCompletionModule
            reconstruction_loss_weight: Weight for reconstruction loss
            cycle_loss_weight: Weight for cycle consistency loss
            loss_type: Type of loss to use
        """
        super().__init__()
        self.completion_module = completion_module
        self.reconstruction_loss_weight = reconstruction_loss_weight
        self.cycle_loss_weight = cycle_loss_weight
        
        self.recon_loss_fn = FeatureReconstructionLoss(loss_type=loss_type)
        self.cycle_loss_fn = CycleConsistencyLoss(loss_type=loss_type)
        
    def compute_reconstruction_loss(self,
                                      real_features: Dict[str, torch.Tensor],
                                      mask_modality: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute reconstruction loss for a masked modality.
        
        Args:
            real_features: Dict of all real modality features
            mask_modality: Name of modality to mask and reconstruct
            
        Returns:
            Tuple of (generated_feature, reconstruction_loss)
        """
        # Create mask with the specified modality missing
        modality_names = self.completion_module.modality_names
        modality_mask = [name != mask_modality for name in modality_names]
        
        # Get available features (excluding masked modality)
        available_features = {
            name: feat for name, feat in real_features.items() 
            if name != mask_modality
        }
        
        # Generate the masked modality's feature
        generated = self.completion_module.generate_missing_features(
            available_features, modality_mask
        )
        
        if mask_modality not in generated:
            return None, torch.tensor(0.0, device=next(iter(real_features.values())).device)
        
        generated_feat = generated[mask_modality]
        real_feat = real_features[mask_modality]
        
        # Compute reconstruction loss
        recon_loss = self.recon_loss_fn(generated_feat, real_feat)
        
        return generated_feat, recon_loss
    
    def compute_cycle_consistency_loss(self,
                                         real_features: Dict[str, torch.Tensor],
                                         source_modality: str,
                                         intermediate_modality: str) -> torch.Tensor:
        """
        Compute cycle consistency loss: source → intermediate → source.
        
        Args:
            real_features: Dict of all real modality features
            source_modality: Starting modality
            intermediate_modality: Intermediate modality to go through
            
        Returns:
            Cycle consistency loss
        """
        modality_names = self.completion_module.modality_names
        
        # Step 1: Generate intermediate from source
        # Mask intermediate, keep source
        mask_step1 = [name != intermediate_modality for name in modality_names]
        available_step1 = {source_modality: real_features[source_modality]}
        
        generated_intermediate = self.completion_module.generate_missing_features(
            available_step1, mask_step1
        )
        
        if intermediate_modality not in generated_intermediate:
            return torch.tensor(0.0, device=real_features[source_modality].device)
        
        # Step 2: Reconstruct source from generated intermediate
        mask_step2 = [name != source_modality for name in modality_names]
        available_step2 = {intermediate_modality: generated_intermediate[intermediate_modality]}
        
        reconstructed_source = self.completion_module.generate_missing_features(
            available_step2, mask_step2
        )
        
        if source_modality not in reconstructed_source:
            return torch.tensor(0.0, device=real_features[source_modality].device)
        
        # Compute cycle consistency loss
        cycle_loss = self.cycle_loss_fn(
            real_features[source_modality],
            reconstructed_source[source_modality]
        )
        
        return cycle_loss
    
    def forward(self,
                real_features: Dict[str, torch.Tensor],
                compute_cycle: bool = True) -> Dict[str, torch.Tensor]:
        """
        Compute all completion-related losses.
        
        Args:
            real_features: Dict of all real modality features (CLS tokens)
            compute_cycle: Whether to compute cycle consistency loss
            
        Returns:
            Dict of losses
        """
        losses = {}
        device = next(iter(real_features.values())).device
        
        # Compute reconstruction loss for each modality
        recon_losses = []
        modality_names = self.completion_module.modality_names
        
        for modality in modality_names:
            if modality in real_features:
                _, recon_loss = self.compute_reconstruction_loss(real_features, modality)
                recon_losses.append(recon_loss)
        
        if recon_losses:
            total_recon_loss = torch.stack(recon_losses).mean()
            losses['completion_recon_loss'] = total_recon_loss * self.reconstruction_loss_weight
        else:
            losses['completion_recon_loss'] = torch.tensor(0.0, device=device)
        
        # Compute cycle consistency loss for selected pairs
        if compute_cycle:
            cycle_losses = []
            # Define important cycle pairs (focus on RGB as target since it's gallery)
            cycle_pairs = [
                ('TEXT', 'RGB'),  # text → rgb → text
                ('NIR', 'RGB'),   # nir → rgb → nir
                ('RGB', 'TEXT'),  # rgb → text → rgb
            ]
            
            for source, intermediate in cycle_pairs:
                if source in real_features and intermediate in real_features:
                    cycle_loss = self.compute_cycle_consistency_loss(
                        real_features, source, intermediate
                    )
                    cycle_losses.append(cycle_loss)
            
            if cycle_losses:
                total_cycle_loss = torch.stack(cycle_losses).mean()
                losses['completion_cycle_loss'] = total_cycle_loss * self.cycle_loss_weight
            else:
                losses['completion_cycle_loss'] = torch.tensor(0.0, device=device)
        
        return losses


class InferenceCompletionHelper:
    """
    Helper class for using completion during inference.
    
    During inference, when modalities are missing, this helper
    generates pseudo-features to complement the real features.
    """
    def __init__(self, completion_module: CrossModalCompletionModule):
        self.completion_module = completion_module
        self.modality_names = completion_module.modality_names
    
    def complete_for_inference(self,
                                available_features: Dict[str, torch.Tensor],
                                modality_mask: List[bool],
                                target_modalities: List[str] = None) -> Dict[str, torch.Tensor]:
        """
        Complete missing modality features for inference.
        
        Args:
            available_features: Dict of available real features
            modality_mask: List indicating which modalities are present
            target_modalities: Optional list of specific modalities to generate
                              If None, generates all missing modalities
                              
        Returns:
            Dict with both real and generated features
        """
        # Get generated features for missing modalities
        generated = self.completion_module.generate_missing_features(
            available_features, modality_mask
        )
        
        # Filter to target modalities if specified
        if target_modalities is not None:
            generated = {k: v for k, v in generated.items() if k in target_modalities}
        
        # Combine real and generated features
        complete = dict(available_features)
        complete.update(generated)
        
        return complete
    
    def get_completed_fusion_features(self,
                                       available_features: Dict[str, torch.Tensor],
                                       modality_mask: List[bool],
                                       fusion_fn) -> torch.Tensor:
        """
        Get fusion features with completed modalities.
        
        Args:
            available_features: Dict of available real features
            modality_mask: List indicating which modalities are present
            fusion_fn: Fusion function to apply to combined features
            
        Returns:
            Fused features including generated modalities
        """
        # Complete missing features
        complete_features = self.complete_for_inference(
            available_features, modality_mask
        )
        
        # Apply fusion (this would typically be the mm_fusion method)
        # The exact implementation depends on how fusion is done
        return complete_features
