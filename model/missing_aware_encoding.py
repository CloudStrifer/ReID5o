"""
Missing-aware Robust Encoding Module

This module implements the core components for handling arbitrary missing modalities in 
multi-modal person re-identification. The key innovation is to transform the missing modality 
problem from a numerical space (zeros) to a semantic space (explicit missing tokens).

Key Components:
1. Missing Token Embedding: Learnable tokens representing "missing" state for each modality
2. Modality Type Embedding: Tokens indicating which modality the input belongs to
3. Missing Mask Embedding: Binary indicators for present/missing status

This allows the Transformer to distinguish between:
- Absent modalities (missing token + mask=0)
- Low-quality/noisy modalities (actual features + mask=1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from typing import List, Dict, Tuple, Optional


class ModalityTypeEmbedding(nn.Module):
    """
    Learnable embeddings for each modality type.
    Helps the model identify which modality the tokens belong to.
    """
    def __init__(self, embed_dim: int, num_modalities: int = 5):
        """
        Args:
            embed_dim: Dimension of the embedding (should match transformer width)
            num_modalities: Number of different modalities (RGB, NIR, CP, SK, TEXT = 5)
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_modalities = num_modalities
        
        # Modality type embeddings: one for each modality
        # Index: 0=RGB, 1=NIR, 2=CP, 3=SK, 4=TEXT
        self.modality_embeddings = nn.Parameter(
            torch.zeros(num_modalities, embed_dim)
        )
        nn.init.normal_(self.modality_embeddings, std=0.02)
        
    def forward(self, modality_idx: int) -> torch.Tensor:
        """
        Get the embedding for a specific modality.
        
        Args:
            modality_idx: Index of the modality (0-4)
            
        Returns:
            Embedding tensor of shape (embed_dim,)
        """
        return self.modality_embeddings[modality_idx]


class MissingTokenEmbedding(nn.Module):
    """
    Learnable missing tokens for each modality.
    When a modality is missing, instead of zeros, we use these learnable tokens
    to explicitly represent the "missing" state in semantic space.
    """
    def __init__(self, embed_dim: int, num_modalities: int = 5, num_tokens: int = 25):
        """
        Args:
            embed_dim: Dimension of the embedding
            num_modalities: Number of different modalities
            num_tokens: Number of tokens per modality (CLS + patches for vision, or sequence length for text)
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_modalities = num_modalities
        self.num_tokens = num_tokens
        
        # Missing token embeddings: one set for each modality
        # Shape: (num_modalities, num_tokens, embed_dim)
        # This allows different modalities to have distinct "missing" representations
        self.missing_tokens = nn.Parameter(
            torch.zeros(num_modalities, num_tokens, embed_dim)
        )
        nn.init.normal_(self.missing_tokens, std=0.02)
        
    def forward(self, modality_idx: int, batch_size: int) -> torch.Tensor:
        """
        Get the missing token embedding for a specific modality.
        
        Args:
            modality_idx: Index of the modality (0-4)
            batch_size: Batch size to expand the tokens
            
        Returns:
            Missing token tensor of shape (batch_size, num_tokens, embed_dim)
        """
        # Get the missing tokens for this modality and expand to batch size
        missing = self.missing_tokens[modality_idx]  # (num_tokens, embed_dim)
        missing = missing.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, num_tokens, embed_dim)
        return missing


class MissingMaskEmbedding(nn.Module):
    """
    Learnable embeddings for missing/present status.
    Provides explicit binary indicators that the Transformer can attend to.
    """
    def __init__(self, embed_dim: int):
        """
        Args:
            embed_dim: Dimension of the embedding
        """
        super().__init__()
        self.embed_dim = embed_dim
        
        # Two embeddings: one for "present" (1) and one for "missing" (0)
        self.mask_embeddings = nn.Parameter(
            torch.zeros(2, embed_dim)
        )
        nn.init.normal_(self.mask_embeddings, std=0.02)
        
    def forward(self, is_present: bool) -> torch.Tensor:
        """
        Get the mask embedding based on presence status.
        
        Args:
            is_present: True if modality is present, False if missing
            
        Returns:
            Mask embedding tensor of shape (embed_dim,)
        """
        idx = 1 if is_present else 0
        return self.mask_embeddings[idx]


class ModalityDropout(nn.Module):
    """
    Modality Dropout Module for training robustness.
    
    During training, randomly drops modalities to create different subset combinations,
    forcing the model to learn robust representations that work with any subset.
    """
    def __init__(self, 
                 num_modalities: int = 5,
                 min_keep: int = 1,
                 dropout_prob: float = 0.5,
                 keep_rgb_prob: float = 0.8):
        """
        Args:
            num_modalities: Total number of modalities (5 for RGB, NIR, CP, SK, TEXT)
            min_keep: Minimum number of modalities to keep (at least 1)
            dropout_prob: Probability of applying dropout to each non-RGB modality
            keep_rgb_prob: Probability of keeping RGB (usually higher since it's the gallery modality)
        """
        super().__init__()
        self.num_modalities = num_modalities
        self.min_keep = min_keep
        self.dropout_prob = dropout_prob
        self.keep_rgb_prob = keep_rgb_prob
        
        # Modality names for logging/debugging
        self.modality_names = ['RGB', 'NIR', 'CP', 'SK', 'TEXT']
        
    def forward(self, training: bool = True) -> List[bool]:
        """
        Generate a random mask indicating which modalities to keep.
        
        Args:
            training: Whether in training mode (dropout only during training)
            
        Returns:
            List of booleans indicating which modalities are present [RGB, NIR, CP, SK, TEXT]
        """
        if not training:
            # During evaluation, keep all modalities
            return [True] * self.num_modalities
        
        # Generate random mask
        mask = []
        
        # RGB has special treatment (gallery modality)
        mask.append(random.random() < self.keep_rgb_prob)
        
        # Other modalities
        for _ in range(1, self.num_modalities):
            mask.append(random.random() > self.dropout_prob)
        
        # Ensure at least min_keep modalities are present
        if sum(mask) < self.min_keep:
            # Randomly select modalities to keep
            indices = list(range(self.num_modalities))
            random.shuffle(indices)
            for idx in indices:
                if not mask[idx]:
                    mask[idx] = True
                    if sum(mask) >= self.min_keep:
                        break
        
        return mask
    
    def sample_specific_subset(self, subset_indices: List[int]) -> List[bool]:
        """
        Generate mask for a specific subset of modalities.
        
        Args:
            subset_indices: List of modality indices to keep
            
        Returns:
            List of booleans indicating which modalities are present
        """
        mask = [False] * self.num_modalities
        for idx in subset_indices:
            if 0 <= idx < self.num_modalities:
                mask[idx] = True
        return mask


class MissingAwareEncoder(nn.Module):
    """
    Missing-aware Robust Encoding Module.
    
    This module wraps the modality embeddings and provides methods to:
    1. Add modality type information to token embeddings
    2. Replace missing modalities with learnable missing tokens
    3. Add missing mask information
    
    The key insight is that instead of using zeros for missing modalities,
    we use explicit learnable "missing tokens" that allow the Transformer to
    understand that a modality is absent (semantic space) rather than just empty (numerical space).
    """
    def __init__(self, 
                 embed_dim: int,
                 num_modalities: int = 5,
                 vision_num_tokens: int = 25,  # 1 CLS + 24 patches for 384x128 images with patch size 16
                 text_num_tokens: int = 77):    # Standard CLIP text length
        """
        Args:
            embed_dim: Dimension of embeddings (512 for CLIP ViT-B/16)
            num_modalities: Number of modalities (5: RGB, NIR, CP, SK, TEXT)
            vision_num_tokens: Number of tokens for vision modalities
            text_num_tokens: Number of tokens for text modality
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_modalities = num_modalities
        self.vision_num_tokens = vision_num_tokens
        self.text_num_tokens = text_num_tokens
        
        # Modality indices
        self.MODALITY_RGB = 0
        self.MODALITY_NIR = 1
        self.MODALITY_CP = 2
        self.MODALITY_SK = 3
        self.MODALITY_TEXT = 4
        
        # Core embedding modules
        self.modality_type_embedding = ModalityTypeEmbedding(embed_dim, num_modalities)
        
        # Separate missing token embeddings for vision and text (different sequence lengths)
        self.vision_missing_tokens = MissingTokenEmbedding(
            embed_dim, num_modalities=4, num_tokens=vision_num_tokens  # RGB, NIR, CP, SK
        )
        self.text_missing_tokens = MissingTokenEmbedding(
            embed_dim, num_modalities=1, num_tokens=text_num_tokens  # TEXT only
        )
        
        self.missing_mask_embedding = MissingMaskEmbedding(embed_dim)
        
        # Modality dropout for training
        self.modality_dropout = ModalityDropout(num_modalities=num_modalities)
        
        # Projection layer to combine all embeddings
        self.fusion_proj = nn.Linear(embed_dim, embed_dim)
        nn.init.xavier_uniform_(self.fusion_proj.weight)
        nn.init.zeros_(self.fusion_proj.bias)
        
    def get_modality_idx(self, modality_name: str) -> int:
        """Get the index for a modality name."""
        modality_map = {'RGB': 0, 'NIR': 1, 'CP': 2, 'SK': 3, 'TEXT': 4}
        return modality_map.get(modality_name.upper(), 0)
    
    def encode_with_modality_info(self, 
                                   tokens: torch.Tensor, 
                                   modality_idx: int,
                                   is_present: bool = True) -> torch.Tensor:
        """
        Add modality type and missing mask information to token embeddings.
        
        Args:
            tokens: Token embeddings of shape (batch_size, seq_len, embed_dim)
            modality_idx: Index of the modality (0-4)
            is_present: Whether this modality is present or missing
            
        Returns:
            Enhanced token embeddings with modality information
        """
        batch_size = tokens.shape[0]
        target_dtype = tokens.dtype
        target_device = tokens.device
        
        # Get modality type embedding and expand to sequence length
        modality_emb = self.modality_type_embedding(modality_idx)  # (embed_dim,)
        modality_emb = modality_emb.to(dtype=target_dtype, device=target_device)
        modality_emb = modality_emb.unsqueeze(0).unsqueeze(0)  # (1, 1, embed_dim)
        modality_emb = modality_emb.expand(batch_size, tokens.shape[1], -1)  # (batch_size, seq_len, embed_dim)
        
        # Get missing mask embedding and expand
        mask_emb = self.missing_mask_embedding(is_present)  # (embed_dim,)
        mask_emb = mask_emb.to(dtype=target_dtype, device=target_device)
        mask_emb = mask_emb.unsqueeze(0).unsqueeze(0)  # (1, 1, embed_dim)
        mask_emb = mask_emb.expand(batch_size, tokens.shape[1], -1)  # (batch_size, seq_len, embed_dim)
        
        # Combine: original tokens + modality type + missing mask
        enhanced_tokens = tokens + modality_emb + mask_emb
        
        return enhanced_tokens
    
    def get_missing_tokens(self, 
                           modality_idx: int, 
                           batch_size: int,
                           is_text: bool = False,
                           dtype: torch.dtype = None,
                           device: torch.device = None) -> torch.Tensor:
        """
        Get learnable missing tokens for a specific modality.
        
        Args:
            modality_idx: Index of the modality
            batch_size: Batch size
            is_text: Whether this is a text modality
            dtype: Target dtype for the output tensor (for FP16 compatibility)
            device: Target device for the output tensor
            
        Returns:
            Missing token embeddings with modality information
        """
        if is_text:
            # Text modality uses separate missing tokens
            missing = self.text_missing_tokens(0, batch_size)  # Always index 0 for text
        else:
            # Vision modalities (RGB=0, NIR=1, CP=2, SK=3)
            missing = self.vision_missing_tokens(modality_idx, batch_size)
        
        # Convert to target dtype if specified
        if dtype is not None:
            missing = missing.to(dtype=dtype)
        if device is not None:
            missing = missing.to(device=device)
        
        # Add modality type and missing mask embeddings
        enhanced_missing = self.encode_with_modality_info(
            missing, modality_idx, is_present=False
        )
        
        return enhanced_missing
    
    def process_modality(self,
                         tokens: Optional[torch.Tensor],
                         modality_idx: int,
                         batch_size: int,
                         is_present: bool = True,
                         is_text: bool = False) -> torch.Tensor:
        """
        Process a modality's tokens, handling both present and missing cases.
        
        Args:
            tokens: Token embeddings if present, None if missing
            modality_idx: Index of the modality
            batch_size: Batch size (needed for missing tokens)
            is_present: Whether this modality is present
            is_text: Whether this is a text modality
            
        Returns:
            Processed token embeddings
        """
        if is_present and tokens is not None:
            # Modality is present: add modality info to actual tokens
            return self.encode_with_modality_info(tokens, modality_idx, is_present=True)
        else:
            # Modality is missing: use learnable missing tokens
            device = self.modality_type_embedding.modality_embeddings.device
            missing_tokens = self.get_missing_tokens(modality_idx, batch_size, is_text)
            return missing_tokens.to(device)


class ConsistencyConstraint(nn.Module):
    """
    Consistency Constraint Module.
    
    Enforces that embeddings obtained under full-modal input and subset input
    should be as close as possible. This reduces subset distribution drift and
    makes the representation space more stable.
    """
    def __init__(self, loss_type: str = 'cosine', temperature: float = 0.1):
        """
        Args:
            loss_type: Type of consistency loss ('l2', 'cosine', 'both')
            temperature: Temperature for cosine similarity (if used)
        """
        super().__init__()
        self.loss_type = loss_type
        self.temperature = temperature
        
    def forward(self, 
                full_modal_embeds: torch.Tensor, 
                subset_embeds: torch.Tensor,
                normalize: bool = True) -> torch.Tensor:
        """
        Compute consistency loss between full-modal and subset embeddings.
        
        Args:
            full_modal_embeds: Embeddings from full modality input (batch_size, embed_dim)
            subset_embeds: Embeddings from subset modality input (batch_size, embed_dim)
            normalize: Whether to normalize embeddings before computing loss
            
        Returns:
            Consistency loss scalar
        """
        if normalize:
            full_modal_embeds = F.normalize(full_modal_embeds, p=2, dim=1)
            subset_embeds = F.normalize(subset_embeds, p=2, dim=1)
        
        if self.loss_type == 'l2':
            # L2 loss: minimize Euclidean distance
            loss = F.mse_loss(subset_embeds, full_modal_embeds)
            
        elif self.loss_type == 'cosine':
            # Cosine similarity loss: maximize similarity (minimize 1 - similarity)
            similarity = F.cosine_similarity(subset_embeds, full_modal_embeds, dim=1)
            loss = (1 - similarity).mean()
            
        elif self.loss_type == 'both':
            # Combine both losses
            l2_loss = F.mse_loss(subset_embeds, full_modal_embeds)
            cosine_loss = (1 - F.cosine_similarity(subset_embeds, full_modal_embeds, dim=1)).mean()
            loss = l2_loss + cosine_loss
            
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return loss


class MissingAwareModalityAssembler(nn.Module):
    """
    Assembles multi-modal tokens with missing-awareness.
    
    This module combines tokens from multiple modalities, handling missing modalities
    by replacing them with learnable missing tokens. The assembled tokens can then
    be fed to the fusion transformer.
    """
    def __init__(self, 
                 embed_dim: int,
                 num_modalities: int = 5,
                 vision_num_tokens: int = 25,
                 text_num_tokens: int = 77):
        """
        Args:
            embed_dim: Embedding dimension
            num_modalities: Number of modalities
            vision_num_tokens: Number of tokens per vision modality
            text_num_tokens: Number of tokens for text modality
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_modalities = num_modalities
        self.vision_num_tokens = vision_num_tokens
        self.text_num_tokens = text_num_tokens
        
        # Missing-aware encoder
        self.missing_encoder = MissingAwareEncoder(
            embed_dim=embed_dim,
            num_modalities=num_modalities,
            vision_num_tokens=vision_num_tokens,
            text_num_tokens=text_num_tokens
        )
        
        # Consistency constraint
        self.consistency_constraint = ConsistencyConstraint(loss_type='cosine')
        
    def assemble_modalities(self,
                           rgb_tokens: Optional[torch.Tensor] = None,
                           nir_tokens: Optional[torch.Tensor] = None,
                           cp_tokens: Optional[torch.Tensor] = None,
                           sk_tokens: Optional[torch.Tensor] = None,
                           text_tokens: Optional[torch.Tensor] = None,
                           modality_mask: List[bool] = None,
                           batch_size: int = None) -> Tuple[torch.Tensor, Dict[str, bool]]:
        """
        Assemble tokens from multiple modalities with missing-awareness.
        
        Args:
            rgb_tokens: RGB modality tokens (batch_size, vision_num_tokens, embed_dim)
            nir_tokens: NIR modality tokens
            cp_tokens: CP modality tokens
            sk_tokens: SK modality tokens
            text_tokens: Text modality tokens (batch_size, text_num_tokens, embed_dim)
            modality_mask: List of booleans [RGB, NIR, CP, SK, TEXT] indicating presence
            batch_size: Batch size (required if all tokens are None)
            
        Returns:
            Tuple of (assembled_tokens, modality_info_dict)
        """
        # Determine batch size
        if batch_size is None:
            for tokens in [rgb_tokens, nir_tokens, cp_tokens, sk_tokens, text_tokens]:
                if tokens is not None:
                    batch_size = tokens.shape[0]
                    break
        
        if batch_size is None:
            raise ValueError("Cannot determine batch size - provide at least one modality or batch_size")
        
        # Default mask: all present
        if modality_mask is None:
            modality_mask = [True] * self.num_modalities
        
        # Process each modality
        all_modality_tokens = [rgb_tokens, nir_tokens, cp_tokens, sk_tokens, text_tokens]
        is_text = [False, False, False, False, True]
        
        processed_tokens = []
        modality_info = {}
        
        for idx, (tokens, mask, text_flag) in enumerate(zip(all_modality_tokens, modality_mask, is_text)):
            modality_name = ['RGB', 'NIR', 'CP', 'SK', 'TEXT'][idx]
            is_present = mask and tokens is not None
            
            processed = self.missing_encoder.process_modality(
                tokens=tokens,
                modality_idx=idx,
                batch_size=batch_size,
                is_present=is_present,
                is_text=text_flag
            )
            
            processed_tokens.append(processed)
            modality_info[modality_name] = is_present
        
        # Concatenate all processed tokens
        # Shape: (batch_size, total_tokens, embed_dim)
        assembled = torch.cat(processed_tokens, dim=1)
        
        return assembled, modality_info
    
    def compute_consistency_loss(self,
                                  full_modal_embed: torch.Tensor,
                                  subset_embed: torch.Tensor) -> torch.Tensor:
        """
        Compute consistency loss between full-modal and subset embeddings.
        
        Args:
            full_modal_embed: Embedding from full modality input
            subset_embed: Embedding from subset modality input
            
        Returns:
            Consistency loss
        """
        return self.consistency_constraint(full_modal_embed, subset_embed)
