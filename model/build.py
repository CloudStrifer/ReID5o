#from model import objectives
from . import objectives
from .clip_model import build_CLIP_from_openai_pretrained, convert_weights, Transformer, LayerNorm ,QuickGELU
import torch
import copy
import itertools
import random
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
from .mmencoder_withlora import MMTransformer_withlora
from .missing_aware_encoding import (
    MissingAwareEncoder, 
    MissingAwareModalityAssembler,
    ConsistencyConstraint,
    ModalityDropout
)
from .cross_modal_completion import (
    CrossModalCompletionModule,
    CrossModalCompletionTrainer,
    InferenceCompletionHelper,
    FeatureReconstructionLoss,
    CycleConsistencyLoss
)
from .reliability_adaptive_fusion import (
    ReliabilityAdaptiveFusion,
    ReliabilityAdaptiveFusionTrainer,
    AdaptiveFusionInferenceHelper
)

######################ReID5o Model########################
class VisionTokenizer(nn.Module):
    def __init__(self, conv1, class_embedding, positional_embedding,ln_pre):
        super(VisionTokenizer, self).__init__()
        self.conv1 = conv1
        self.class_embedding = class_embedding
        self.positional_embedding = positional_embedding
        self.ln_pre = ln_pre

    def forward(self, x):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        return x


class MultimodalVisionEncoder(nn.Module):
    def __init__(self, transformer, ln_post, proj):
        super(MultimodalVisionEncoder, self).__init__()
        self.transformer = transformer
        self.ln_post = ln_post
        self.proj = proj

    def forward(self, x,modality='RGB'):
        x = x.permute(1, 0, 2)
        x = self.transformer(x,modality)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x)
        x = x @ self.proj
        return x


class CLIPTextEncoder(nn.Module):
    def __init__(self, token_embedding, positional_embedding,transformer,ln_final,text_projection):
        super(CLIPTextEncoder, self).__init__()
        self.token_embedding = token_embedding
        self.positional_embedding = positional_embedding
        self.transformer = transformer
        self.ln_final = ln_final
        self.text_projection = text_projection

    def forward(self, text, dtype):
        x = self.token_embedding(text).type(dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x @ self.text_projection
        eot_embed = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
        embeds = x
        #x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return eot_embed, embeds


class ReID5oModel(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()
        base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size,args.stride_size)
        self.embed_dim = base_cfg['embed_dim']  # 512
        self.width = base_cfg['vision_width']  # 768
        self.image_resolution = base_cfg['image_resolution']
        self.encoder_layers = base_cfg['vision_layers']
        self.heads = self.embed_dim //64

        self.mmt_depth = args.mmt_depth

        self.clip_text_encoder = self.build_clip_pretrained_text_encoder(base_model)
        self.rgb_tokenizer = self.build_vision_tokenizer(base_model)
        self.nir_tokenizer = self.build_vision_tokenizer(base_model)
        self.cp_tokenizer = self.build_vision_tokenizer(base_model)
        self.sk_tokenizer = self.build_vision_tokenizer(base_model)
        self.vision_encoder = self.build_vision_encoder(base_model,args)

        self.logit_scale = torch.ones([]) * (1 / args.temperature)

        if 'mm' in args.loss_names:
            self.create_mm_fusion_module()

        if 'id' in args.loss_names:
            self.create_id_classifier()
        
        # ============ Missing-aware Robust Encoding ============
        # Initialize missing-aware encoding components
        self.use_missing_aware = getattr(args, 'use_missing_aware', False)
        
        # Check if using 3-modal dataset (CUHK-PEDES, ICFG-PEDES, RSTPReid)
        # Force NIR and CP to be treated as always missing
        dataset_name = getattr(args, 'dataset_name', 'ORBench')
        self.force_missing_nir_cp = getattr(args, 'force_missing_nir_cp', False)
        three_modal_datasets = [
            'CUHK-PEDES', 'CUHK_PEDES', 'CUHK-PEDES-3Modal',
            'ICFG-PEDES', 'ICFG_PEDES', 'ICFG-PEDES-3Modal',
            'RSTPReid', 'RSTPReid-3Modal'
        ]
        if dataset_name in three_modal_datasets:
            self.force_missing_nir_cp = True
        
        if self.use_missing_aware:
            self._init_missing_aware_encoding(args)
        
        # ============ Cross-modal Feature Completion ============
        # Initialize cross-modal completion components
        self.use_cross_modal_completion = getattr(args, 'use_cross_modal_completion', False)
        if self.use_cross_modal_completion:
            self._init_cross_modal_completion(args)
        
        # ============ Reliability-Adaptive Fusion ============
        # Initialize reliability-adaptive fusion components
        self.use_reliability_fusion = getattr(args, 'use_reliability_fusion', False)
        if self.use_reliability_fusion:
            self._init_reliability_adaptive_fusion(args)

    def create_id_classifier(self):
        print('num_classes:{}'.format(self.num_classes))
        self.classifier = nn.Linear(self.embed_dim, self.num_classes)
        nn.init.normal_(self.classifier.weight.data, std=0.001)
        nn.init.constant_(self.classifier.bias.data, val=0.0)

    def _init_missing_aware_encoding(self, args):
        """
        Initialize Missing-aware Robust Encoding components.
        
        This enables the model to handle arbitrary missing modalities by:
        1. Using learnable missing tokens instead of zeros
        2. Adding modality type embeddings
        3. Adding missing mask embeddings
        4. Applying modality dropout during training
        """
        print('Initializing Missing-aware Robust Encoding...')
        
        # Calculate vision token count: 1 CLS + grid patches
        # For 384x128 image with patch size 16: 24x8 = 192 patches, but after reshape: (384/16) * (128/16) = 24 * 8 = 192
        # Actually for CLIP: grid_h = 384/16 = 24, grid_w = 128/16 = 8, total = 24*8 = 192 + 1 CLS = 193
        # But based on the code, it seems to use a different calculation
        grid_h = args.img_size[0] // args.stride_size
        grid_w = args.img_size[1] // args.stride_size
        self.vision_num_tokens = grid_h * grid_w + 1  # +1 for CLS token
        self.text_num_tokens = args.text_length  # 77
        
        print(f'Vision tokens: {self.vision_num_tokens}, Text tokens: {self.text_num_tokens}')
        
        # Check if using 3-modal dataset (CUHK-PEDES, ICFG-PEDES, RSTPReid)
        dataset_name = getattr(args, 'dataset_name', 'ORBench')
        three_modal_datasets = [
            'CUHK-PEDES', 'CUHK_PEDES', 'CUHK-PEDES-3Modal',
            'ICFG-PEDES', 'ICFG_PEDES', 'ICFG-PEDES-3Modal',
            'RSTPReid', 'RSTPReid-3Modal'
        ]
        if dataset_name in three_modal_datasets:
            self.force_missing_nir_cp = True
            print(f'Detected 3-modal dataset ({dataset_name}): forcing NIR and CP as always missing')
        
        # Missing-aware encoder for handling missing modalities
        self.missing_aware_encoder = MissingAwareEncoder(
            embed_dim=self.embed_dim,
            num_modalities=5,  # RGB, NIR, CP, SK, TEXT
            vision_num_tokens=self.vision_num_tokens,
            text_num_tokens=self.text_num_tokens
        )
        
        # Modality dropout for training robustness
        self.modality_dropout = ModalityDropout(
            num_modalities=5,
            min_keep=getattr(args, 'min_modalities_keep', 1),
            dropout_prob=getattr(args, 'modality_dropout_prob', 0.5),
            keep_rgb_prob=getattr(args, 'keep_rgb_prob', 0.8)
        )
        
        # Consistency constraint for aligning full-modal and subset embeddings
        self.consistency_constraint = ConsistencyConstraint(
            loss_type=getattr(args, 'consistency_loss_type', 'cosine'),
            temperature=getattr(args, 'consistency_temperature', 0.1)
        )
        
        # Weight for consistency loss
        self.consistency_loss_weight = getattr(args, 'consistency_loss_weight', 0.1)
        
        # Warmup epoch for modality dropout (train normally first, then introduce dropout)
        self.modality_dropout_warmup_epochs = getattr(args, 'modality_dropout_warmup_epochs', 5)
        
        print(f'Missing-aware Robust Encoding initialized with:')
        print(f'  - Modality dropout prob: {getattr(args, "modality_dropout_prob", 0.5)}')
        print(f'  - Keep RGB prob: {getattr(args, "keep_rgb_prob", 0.8)}')
        print(f'  - Min modalities keep: {getattr(args, "min_modalities_keep", 1)}')
        print(f'  - Consistency loss weight: {self.consistency_loss_weight}')

    def _init_cross_modal_completion(self, args):
        """
        Initialize Cross-modal Feature Completion components.
        
        This module generates pseudo-features for missing modalities using
        features from available modalities. It enables "information recovery"
        rather than just surviving with missing modalities.
        """
        print('Initializing Cross-modal Feature Completion...')
        
        # Cross-modal completion module
        self.completion_module = CrossModalCompletionModule(
            embed_dim=self.embed_dim,
            num_modalities=5,  # RGB, NIR, CP, SK, TEXT
            num_heads=getattr(args, 'completion_num_heads', 8),
            num_layers=getattr(args, 'completion_num_layers', 2),
            dropout=getattr(args, 'completion_dropout', 0.1)
        )
        
        # Training helper for completion
        self.completion_trainer = CrossModalCompletionTrainer(
            completion_module=self.completion_module,
            reconstruction_loss_weight=getattr(args, 'completion_recon_loss_weight', 1.0),
            cycle_loss_weight=getattr(args, 'completion_cycle_loss_weight', 0.5),
            loss_type=getattr(args, 'completion_loss_type', 'cosine')
        )
        
        # Inference helper
        self.completion_inference = InferenceCompletionHelper(self.completion_module)
        
        # Whether to use completion during inference
        self.use_completion_inference = getattr(args, 'use_completion_inference', True)
        
        # Store weights
        self.completion_recon_loss_weight = getattr(args, 'completion_recon_loss_weight', 1.0)
        self.completion_cycle_loss_weight = getattr(args, 'completion_cycle_loss_weight', 0.5)
        
        print(f'Cross-modal Feature Completion initialized with:')
        print(f'  - Num heads: {getattr(args, "completion_num_heads", 8)}')
        print(f'  - Num layers: {getattr(args, "completion_num_layers", 2)}')
        print(f'  - Reconstruction loss weight: {self.completion_recon_loss_weight}')
        print(f'  - Cycle loss weight: {self.completion_cycle_loss_weight}')
        print(f'  - Use completion during inference: {self.use_completion_inference}')

    def _init_reliability_adaptive_fusion(self, args):
        """
        Initialize Reliability-Adaptive Fusion components.
        
        This module dynamically adjusts fusion weights based on modality reliability.
        It enables intelligent modality selection by:
        1. Estimating reliability scores for each modality
        2. Computing quality indicators (variance, norm, confidence)
        3. Applying adaptive weighting based on reliability
        4. Using sparsity regularization and uncertainty-aware loss
        """
        print('Initializing Reliability-Adaptive Fusion...')
        
        # Main fusion module
        self.reliability_fusion = ReliabilityAdaptiveFusion(
            embed_dim=self.embed_dim,
            num_modalities=5,  # RGB, NIR, CP, SK, TEXT
            hidden_dim=getattr(args, 'reliability_hidden_dim', 256),
            num_heads=getattr(args, 'reliability_num_heads', 8),
            num_fusion_layers=getattr(args, 'reliability_num_layers', 2),
            use_quality_indicators=getattr(args, 'use_quality_indicators', True),
            use_transformer_refinement=getattr(args, 'use_transformer_refinement', True)
        )
        
        # Training helper with regularization losses
        self.reliability_fusion_trainer = ReliabilityAdaptiveFusionTrainer(
            fusion_module=self.reliability_fusion,
            sparsity_weight=getattr(args, 'fusion_sparsity_weight', 0.1),
            uncertainty_weight=getattr(args, 'fusion_uncertainty_weight', 0.2),
            sparsity_target=getattr(args, 'fusion_sparsity_target', 0.3),
            sparsity_type=getattr(args, 'fusion_sparsity_type', 'entropy')
        )
        
        # Inference helper
        self.reliability_fusion_inference = AdaptiveFusionInferenceHelper(self.reliability_fusion)
        
        # Whether to use reliability fusion during inference
        self.use_reliability_fusion_inference = getattr(args, 'use_reliability_fusion_inference', True)
        
        # Store weights
        self.fusion_sparsity_weight = getattr(args, 'fusion_sparsity_weight', 0.1)
        self.fusion_uncertainty_weight = getattr(args, 'fusion_uncertainty_weight', 0.2)
        
        print(f'Reliability-Adaptive Fusion initialized with:')
        print(f'  - Hidden dim: {getattr(args, "reliability_hidden_dim", 256)}')
        print(f'  - Num heads: {getattr(args, "reliability_num_heads", 8)}')
        print(f'  - Num layers: {getattr(args, "reliability_num_layers", 2)}')
        print(f'  - Use quality indicators: {getattr(args, "use_quality_indicators", True)}')
        print(f'  - Use transformer refinement: {getattr(args, "use_transformer_refinement", True)}')
        print(f'  - Sparsity weight: {self.fusion_sparsity_weight}')
        print(f'  - Uncertainty weight: {self.fusion_uncertainty_weight}')
        print(f'  - Use during inference: {self.use_reliability_fusion_inference}')

    def create_mm_fusion_module(self):
        self.cross_attn = nn.MultiheadAttention(self.embed_dim,
                                                self.embed_dim // 64,
                                                batch_first=True)
        self.cross_modal_transformer = Transformer(width=self.embed_dim,
                                                   layers=self.mmt_depth,
                                                   heads=self.embed_dim //
                                                         64)
        scale = self.cross_modal_transformer.width ** -0.5

        self.ln_pre = LayerNorm(self.embed_dim)
        self.ln_post = LayerNorm(self.embed_dim)

        proj_std = scale * ((2 * self.cross_modal_transformer.layers) ** -0.5)
        attn_std = scale
        fc_std = (2 * self.cross_modal_transformer.width) ** -0.5
        for block in self.cross_modal_transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        # init cross attn
        nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
        nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

        self.mm_head = nn.Sequential(
            OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                         ('gelu', QuickGELU()),
                         ('ln', LayerNorm(self.embed_dim)),
                         ('fc', nn.Linear(self.embed_dim, self.embed_dim))]))
        # init mlm head
        nn.init.normal_(self.mm_head.dense.weight, std=fc_std)
        nn.init.normal_(self.mm_head.fc.weight, std=proj_std)

    def mm_fusion(self, q, k, v):
        x = self.cross_attn(self.ln_pre(q),self.ln_pre(k),self.ln_pre(v),need_weights=False)[0]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.cross_modal_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x)
        x = self.mm_head(x)
        x =  torch.mean(x,dim=1).float()
        return x

    @property
    def dtype(self):
        return self.rgb_tokenizer.conv1.weight.dtype

    def build_vision_tokenizer(self,base_model):
        conv1 = copy.deepcopy(base_model.visual.conv1)
        cls = copy.deepcopy(base_model.visual.class_embedding)
        pe = copy.deepcopy(base_model.visual.positional_embedding)
        ln_pre = copy.deepcopy(base_model.visual.ln_pre)
        return VisionTokenizer(conv1, cls, pe, ln_pre)

    def build_vision_encoder(self,base_model,args):
        if args.add_lora:
            transformer = MMTransformer_withlora(width=self.width,layers=self.encoder_layers,heads=self.heads,lora_r=args.lora_r, num_loras=args.num_loras,lora_layers=args.lora_layers,lora_mode=args.lora_mode)
            stat = copy.deepcopy(base_model.visual.transformer).state_dict()
            transformer.load_state_dict(stat,strict=False)
            print('Pretrained Multimodal Encoder with LoRAs Loaded, with LoRA_r={}, LoRA_layers={}'.format(args.lora_r,args.lora_layers))
        else:
            transformer = copy.deepcopy(base_model.visual.transformer)
        ln_post = copy.deepcopy(base_model.visual.ln_post)
        proj = copy.deepcopy(base_model.visual.proj)
        encoder = MultimodalVisionEncoder(transformer,ln_post,proj)
        return encoder

    def build_clip_pretrained_text_encoder(self,base_model):
        transformer = copy.deepcopy(base_model.transformer)
        token_embedding = copy.deepcopy(base_model.token_embedding)
        positional_embedding = copy.deepcopy(base_model.positional_embedding)
        ln_final = copy.deepcopy(base_model.ln_final)
        text_projection = copy.deepcopy(base_model.text_projection)
        return CLIPTextEncoder(token_embedding, positional_embedding,transformer,ln_final,text_projection)

    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')

    def encode_rgb_cls(self,x):
        x = self.rgb_tokenizer(x.type(self.dtype))
        x = self.vision_encoder(x,'RGB')
        return x[:, 0, :].float()

    def encode_nir_cls(self,x):
        x = self.nir_tokenizer(x.type(self.dtype))
        x = self.vision_encoder(x,'NIR')
        return x[:, 0, :].float()

    def encode_cp_cls(self,x):
        x = self.cp_tokenizer(x.type(self.dtype))
        x = self.vision_encoder(x,'CP')
        return x[:, 0, :].float()

    def encode_sk_cls(self,x):
        x = self.sk_tokenizer(x.type(self.dtype))
        x = self.vision_encoder(x,'SK')
        return x[:, 0, :].float()

    def encode_text_cls(self,x):
        x,_ = self.clip_text_encoder(x,self.dtype)
        x = x.float()
        return x

    def encode_rgb_embeds(self,x):
        x = self.rgb_tokenizer(x.type(self.dtype))
        x = self.vision_encoder(x,'RGB')
        return x

    def encode_nir_embeds(self,x):
        x = self.nir_tokenizer(x.type(self.dtype))
        x = self.vision_encoder(x,'NIR')
        return x

    def encode_cp_embeds(self,x):
        x = self.cp_tokenizer(x.type(self.dtype))
        x = self.vision_encoder(x,'CP')
        return x

    def encode_sk_embeds(self,x):
        x = self.sk_tokenizer(x.type(self.dtype))
        x = self.vision_encoder(x,'SK')
        return x

    def encode_text_embeds(self,x):
        eot,x = self.clip_text_encoder(x,self.dtype)
        return eot,x

    # ============ Missing-aware Encoding Methods ============
    
    def encode_rgb_embeds_with_missing_aware(self, x, is_present=True):
        """Encode RGB with missing-aware information."""
        if is_present and x is not None:
            embeds = self.encode_rgb_embeds(x)
            if self.use_missing_aware:
                embeds = self.missing_aware_encoder.encode_with_modality_info(
                    embeds, modality_idx=0, is_present=True
                )
            return embeds
        else:
            # Return missing tokens for RGB
            batch_size = x.shape[0] if x is not None else 1
            device = x.device if x is not None else next(self.parameters()).device
            return self.missing_aware_encoder.get_missing_tokens(
                modality_idx=0, batch_size=batch_size, is_text=False,
                dtype=self.dtype, device=device
            )
    
    def encode_nir_embeds_with_missing_aware(self, x, is_present=True):
        """Encode NIR with missing-aware information."""
        if is_present and x is not None:
            embeds = self.encode_nir_embeds(x)
            if self.use_missing_aware:
                embeds = self.missing_aware_encoder.encode_with_modality_info(
                    embeds, modality_idx=1, is_present=True
                )
            return embeds
        else:
            batch_size = x.shape[0] if x is not None else 1
            device = x.device if x is not None else next(self.parameters()).device
            return self.missing_aware_encoder.get_missing_tokens(
                modality_idx=1, batch_size=batch_size, is_text=False,
                dtype=self.dtype, device=device
            )
    
    def encode_cp_embeds_with_missing_aware(self, x, is_present=True):
        """Encode CP with missing-aware information."""
        if is_present and x is not None:
            embeds = self.encode_cp_embeds(x)
            if self.use_missing_aware:
                embeds = self.missing_aware_encoder.encode_with_modality_info(
                    embeds, modality_idx=2, is_present=True
                )
            return embeds
        else:
            batch_size = x.shape[0] if x is not None else 1
            device = x.device if x is not None else next(self.parameters()).device
            return self.missing_aware_encoder.get_missing_tokens(
                modality_idx=2, batch_size=batch_size, is_text=False,
                dtype=self.dtype, device=device
            )
    
    def encode_sk_embeds_with_missing_aware(self, x, is_present=True):
        """Encode SK with missing-aware information."""
        if is_present and x is not None:
            embeds = self.encode_sk_embeds(x)
            if self.use_missing_aware:
                embeds = self.missing_aware_encoder.encode_with_modality_info(
                    embeds, modality_idx=3, is_present=True
                )
            return embeds
        else:
            batch_size = x.shape[0] if x is not None else 1
            device = x.device if x is not None else next(self.parameters()).device
            return self.missing_aware_encoder.get_missing_tokens(
                modality_idx=3, batch_size=batch_size, is_text=False,
                dtype=self.dtype, device=device
            )
    
    def encode_text_embeds_with_missing_aware(self, x, is_present=True):
        """Encode Text with missing-aware information."""
        if is_present and x is not None:
            eot, embeds = self.encode_text_embeds(x)
            if self.use_missing_aware:
                embeds = self.missing_aware_encoder.encode_with_modality_info(
                    embeds, modality_idx=4, is_present=True
                )
            return eot, embeds
        else:
            batch_size = x.shape[0] if x is not None else 1
            device = x.device if x is not None else next(self.parameters()).device
            missing_embeds = self.missing_aware_encoder.get_missing_tokens(
                modality_idx=4, batch_size=batch_size, is_text=True,
                dtype=self.dtype, device=device
            )
            # For missing text, use mean of missing tokens as EOT
            missing_eot = missing_embeds.mean(dim=1)
            return missing_eot, missing_embeds
    
    def sample_modality_dropout(self):
        """
        Sample a modality dropout mask for training.
        Returns a list of booleans [RGB, NIR, CP, SK, TEXT] indicating which modalities to use.
        """
        if self.use_missing_aware and self.training:
            mask = self.modality_dropout(training=True)
            # For 3-modal datasets, always mark NIR and CP as missing
            if getattr(self, 'force_missing_nir_cp', False):
                mask[1] = False  # NIR
                mask[2] = False  # CP
            return mask
        
        # During evaluation, check for 3-modal datasets
        if getattr(self, 'force_missing_nir_cp', False):
            return [True, False, False, True, True]  # RGB, SK, TEXT present; NIR, CP missing
        return [True, True, True, True, True]  # All present during evaluation
    
    def router_multimodal_embeds_with_missing_aware(self, rgb, nir, cp, sk, text, modality_mask=None):
        """
        Route multi-modal embeddings with missing-awareness.
        
        When modality_mask is provided, missing modalities are replaced with 
        learnable missing tokens instead of being omitted.
        
        Args:
            rgb, nir, cp, sk, text: Input tensors for each modality
            modality_mask: List of booleans [RGB, NIR, CP, SK, TEXT] indicating presence
            
        Returns:
            Same structure as router_multimodal_embeds but with missing-aware handling
        """
        if modality_mask is None:
            modality_mask = [True, True, True, True, True]
        
        # Get batch size
        batch_size = rgb.shape[0]
        
        # Encode each modality with missing-awareness
        if modality_mask[0]:  # RGB
            rgb_embeds = self.encode_rgb_embeds_with_missing_aware(rgb, is_present=True)
        else:
            rgb_embeds = self.encode_rgb_embeds_with_missing_aware(rgb, is_present=False)
        
        if modality_mask[1]:  # NIR
            nir_embeds = self.encode_nir_embeds_with_missing_aware(nir, is_present=True)
        else:
            nir_embeds = self.encode_nir_embeds_with_missing_aware(nir, is_present=False)
        
        if modality_mask[2]:  # CP
            cp_embeds = self.encode_cp_embeds_with_missing_aware(cp, is_present=True)
        else:
            cp_embeds = self.encode_cp_embeds_with_missing_aware(cp, is_present=False)
        
        if modality_mask[3]:  # SK
            sk_embeds = self.encode_sk_embeds_with_missing_aware(sk, is_present=True)
        else:
            sk_embeds = self.encode_sk_embeds_with_missing_aware(sk, is_present=False)
        
        if modality_mask[4]:  # TEXT
            text_eot, text_embeds = self.encode_text_embeds_with_missing_aware(text, is_present=True)
        else:
            text_eot, text_embeds = self.encode_text_embeds_with_missing_aware(text, is_present=False)
        
        # Ensure all embeddings are in the model's dtype (half precision)
        # This is critical for compatibility with CLIP's FP16 model
        rgb_embeds = rgb_embeds.to(self.dtype)
        nir_embeds = nir_embeds.to(self.dtype)
        cp_embeds = cp_embeds.to(self.dtype)
        sk_embeds = sk_embeds.to(self.dtype)
        text_embeds = text_embeds.to(self.dtype)
        text_eot = text_eot.to(self.dtype)
        
        # For single modality embeddings, use CLS token (index 0)
        mm_embeds = [nir_embeds, cp_embeds, sk_embeds, text_embeds]
        combined_embeds_for_one = [
            rgb_embeds[:, 0, :].float(),
            nir_embeds[:, 0, :].float(),
            cp_embeds[:, 0, :].float(),
            sk_embeds[:, 0, :].float(),
            text_eot.float()
        ]
        
        # Generate combinations for two, three, and four modalities
        combinations_for_two = itertools.combinations(mm_embeds, 2)
        combined_embeds_for_two = []
        for combo in combinations_for_two:
            combined = torch.cat(combo, dim=1)
            combined_embeds_for_two.append(combined)
        
        combinations_for_three = itertools.combinations(mm_embeds, 3)
        combined_embeds_for_three = []
        for combo in combinations_for_three:
            combined = torch.cat(combo, dim=1)
            combined_embeds_for_three.append(combined)
        
        combined_embeds_for_four = [torch.cat(mm_embeds, dim=1)]
        
        return combined_embeds_for_one, combined_embeds_for_two, combined_embeds_for_three, combined_embeds_for_four
    
    def compute_consistency_loss(self, full_modal_embeds, subset_embeds_list):
        """
        Compute consistency loss between full-modal and subset embeddings.
        
        Args:
            full_modal_embeds: Embeddings from full modality input (batch_size, embed_dim)
            subset_embeds_list: List of embeddings from different subsets
            
        Returns:
            Consistency loss scalar
        """
        if not self.use_missing_aware:
            return torch.tensor(0.0, device=full_modal_embeds.device)
        
        total_loss = 0.0
        for subset_embeds in subset_embeds_list:
            total_loss += self.consistency_constraint(full_modal_embeds, subset_embeds)
        
        return total_loss / len(subset_embeds_list) if subset_embeds_list else torch.tensor(0.0)

    # ============ Cross-modal Feature Completion Methods ============
    
    def extract_cls_features(self, rgb, nir, cp, sk, text) -> Dict[str, torch.Tensor]:
        """
        Extract CLS-level features for all modalities.
        Used for cross-modal completion training and inference.
        
        Returns:
            Dict mapping modality names to CLS features (batch_size, embed_dim)
        """
        features = {}
        
        # Encode each modality and get CLS token
        rgb_embeds = self.encode_rgb_embeds(rgb)
        features['RGB'] = rgb_embeds[:, 0, :].float()
        
        nir_embeds = self.encode_nir_embeds(nir)
        features['NIR'] = nir_embeds[:, 0, :].float()
        
        cp_embeds = self.encode_cp_embeds(cp)
        features['CP'] = cp_embeds[:, 0, :].float()
        
        sk_embeds = self.encode_sk_embeds(sk)
        features['SK'] = sk_embeds[:, 0, :].float()
        
        text_eot, _ = self.encode_text_embeds(text)
        features['TEXT'] = text_eot.float()
        
        return features
    
    def compute_completion_losses(self, 
                                   modality_features: Dict[str, torch.Tensor],
                                   compute_cycle: bool = True) -> Dict[str, torch.Tensor]:
        """
        Compute cross-modal completion losses for training.
        
        Args:
            modality_features: Dict of real modality features (CLS tokens)
            compute_cycle: Whether to compute cycle consistency loss
            
        Returns:
            Dict of completion-related losses
        """
        if not self.use_cross_modal_completion:
            device = next(iter(modality_features.values())).device
            return {
                'completion_recon_loss': torch.tensor(0.0, device=device),
                'completion_cycle_loss': torch.tensor(0.0, device=device)
            }
        
        return self.completion_trainer(modality_features, compute_cycle=compute_cycle)
    
    def complete_missing_features(self,
                                    available_features: Dict[str, torch.Tensor],
                                    modality_mask: List[bool]) -> Dict[str, torch.Tensor]:
        """
        Generate pseudo-features for missing modalities during inference.
        
        Args:
            available_features: Dict of available real features
            modality_mask: List indicating which modalities are present
            
        Returns:
            Complete feature dict with both real and generated features
        """
        if not self.use_cross_modal_completion:
            return available_features
        
        return self.completion_inference.complete_for_inference(
            available_features, modality_mask
        )
    
    def get_completed_single_modality_features(self,
                                                 modality_features: Dict[str, torch.Tensor],
                                                 modality_mask: List[bool]) -> List[torch.Tensor]:
        """
        Get single-modality features with completion for missing modalities.
        
        During inference, if a modality is missing, we generate its feature
        from available modalities.
        
        Args:
            modality_features: Dict of available features
            modality_mask: List indicating which modalities are present
            
        Returns:
            List of features [RGB, NIR, CP, SK, TEXT] with generated features for missing ones
        """
        if not self.use_cross_modal_completion or not self.use_completion_inference:
            # Return available features, None for missing
            modality_names = ['RGB', 'NIR', 'CP', 'SK', 'TEXT']
            return [
                modality_features.get(name, None) 
                for name, present in zip(modality_names, modality_mask)
            ]
        
        # Complete missing features
        complete_features = self.complete_missing_features(modality_features, modality_mask)
        
        modality_names = ['RGB', 'NIR', 'CP', 'SK', 'TEXT']
        return [complete_features.get(name) for name in modality_names]

    def compute_reliability_fusion(self,
                                    modality_features: Dict[str, torch.Tensor],
                                    is_generated: Dict[str, bool] = None,
                                    return_losses: bool = True) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute reliability-adaptive fusion of modality features.
        
        This method dynamically weighs modalities based on their estimated reliability,
        penalizing low-quality or generated (completed) modalities.
        
        Args:
            modality_features: Dict of modality features (real or generated)
            is_generated: Dict indicating which features are generated (vs real)
            return_losses: Whether to return regularization losses (for training)
            
        Returns:
            Tuple of:
                - fused_features: Adaptively fused features [B, embed_dim]
                - losses: Dict of regularization losses (sparsity, uncertainty)
        """
        if not self.use_reliability_fusion:
            # Fallback to simple averaging
            available_feats = [feat for feat in modality_features.values() if feat is not None]
            if len(available_feats) == 0:
                device = next(self.parameters()).device
                return torch.zeros(1, self.embed_dim, device=device), {}
            stacked = torch.stack(available_feats, dim=1)
            fused = stacked.mean(dim=1)
            return fused, {}
        
        if is_generated is None:
            is_generated = {name: False for name in modality_features.keys()}
        
        if self.training and return_losses:
            # Use trainer for training with regularization losses
            # Trainer returns (fused_features, losses, info) - we discard info here
            fused_features, losses, _ = self.reliability_fusion_trainer(
                modality_features=modality_features,
                is_generated=is_generated
            )
        else:
            # Use main module for inference (no losses needed)
            fused_features, info = self.reliability_fusion(
                modality_features=modality_features,
                is_generated=is_generated,
                return_weights=True
            )
            losses = {}
        
        return fused_features, losses

    def compute_reliability_fusion_with_completion(self,
                                                    real_features: Dict[str, torch.Tensor],
                                                    modality_mask: List[bool],
                                                    return_losses: bool = True) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute reliability-adaptive fusion, including generated features for missing modalities.
        
        This combines cross-modal completion with reliability fusion:
        1. Generate pseudo-features for missing modalities
        2. Track which features are generated vs real
        3. Apply reliability-adaptive fusion with appropriate penalties
        
        Args:
            real_features: Dict of available real features
            modality_mask: List indicating which modalities are present [RGB, NIR, CP, SK, TEXT]
            return_losses: Whether to return regularization losses
            
        Returns:
            Tuple of:
                - fused_features: Adaptively fused features [B, embed_dim]
                - losses: Dict of regularization losses
        """
        modality_names = ['RGB', 'NIR', 'CP', 'SK', 'TEXT']
        
        # Track which features are generated
        is_generated = {name: not present for name, present in zip(modality_names, modality_mask)}
        
        # Get complete features (real + generated for missing)
        if self.use_cross_modal_completion and self.use_completion_inference:
            complete_features = self.complete_missing_features(real_features, modality_mask)
        else:
            # Only use real features
            complete_features = real_features.copy()
            # Mark missing as None
            for name, present in zip(modality_names, modality_mask):
                if not present:
                    complete_features[name] = None
        
        # Filter out None features for fusion
        fusion_features = {k: v for k, v in complete_features.items() if v is not None}
        fusion_is_generated = {k: is_generated[k] for k in fusion_features.keys()}
        
        return self.compute_reliability_fusion(
            modality_features=fusion_features,
            is_generated=fusion_is_generated,
            return_losses=return_losses
        )

    def get_reliability_fused_feature(self,
                                       modality_features: Dict[str, torch.Tensor],
                                       is_generated: Dict[str, bool] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Get reliability-fused feature with analysis (for inference visualization).
        
        Args:
            modality_features: Dict of modality features
            is_generated: Dict indicating which features are generated
            
        Returns:
            Tuple of:
                - fused_features: Adaptively fused features
                - analysis: Dict with reliability scores, weights, dominant modality
        """
        if not self.use_reliability_fusion:
            available_feats = [feat for feat in modality_features.values() if feat is not None]
            if len(available_feats) == 0:
                device = next(self.parameters()).device
                return torch.zeros(1, self.embed_dim, device=device), {}
            stacked = torch.stack(available_feats, dim=1)
            fused = stacked.mean(dim=1)
            return fused, {'method': 'simple_average'}
        
        if is_generated is None:
            is_generated = {name: False for name in modality_features.keys()}
        
        return self.reliability_fusion_inference.fuse_with_analysis(
            modality_features=modality_features,
            is_generated=is_generated
        )

    def router_multimodal_embeds(self,rgb,nir,cp,sk,text):
        rgb_embeds = self.encode_rgb_embeds(rgb)
        nir_embeds = self.encode_nir_embeds(nir)
        cp_embeds = self.encode_cp_embeds(cp)
        sk_embeds = self.encode_sk_embeds(sk)
        text_eot,text_embeds = self.encode_text_embeds(text)
        mm_embeds = [nir_embeds,cp_embeds,sk_embeds,text_embeds]
        combined_embeds_for_one = [rgb_embeds[:,0,:].float(),nir_embeds[:,0,:].float(),cp_embeds[:,0,:].float(),sk_embeds[:,0,:].float(),text_eot.float()]

        combinations_for_two = itertools.combinations(mm_embeds, 2)
        combined_embeds_for_two = []
        for combo in combinations_for_two:
            combined = torch.cat(combo, dim=1)
            combined_embeds_for_two.append(combined)

        combinations_for_three = itertools.combinations(mm_embeds, 3)
        combined_embeds_for_three = []
        for combo in combinations_for_three:
            combined = torch.cat(combo, dim=1)
            combined_embeds_for_three.append(combined)

        combined_embeds_for_four = [torch.cat(mm_embeds,dim=1)]

        return combined_embeds_for_one,combined_embeds_for_two,combined_embeds_for_three,combined_embeds_for_four

    def forward(self, batch, use_modality_dropout=None, current_epoch=None):
        ret = dict()
        rgbs = batch['rgbs']
        nirs = batch['nirs']
        cps = batch['cps']
        sks = batch['sks']
        texts = batch['caption_ids']
        logit_scale = self.logit_scale
        ret.update({'temperature': 1 / logit_scale})
        
        # Determine whether to use modality dropout
        # Check warmup: don't use modality dropout during initial epochs
        modality_dropout_warmup = getattr(self.args, 'modality_dropout_warmup_epochs', 5)
        in_warmup = current_epoch is not None and current_epoch <= modality_dropout_warmup
        
        if use_modality_dropout is None:
            # Only use modality dropout if: training, missing-aware enabled, and past warmup
            use_modality_dropout = self.training and self.use_missing_aware and not in_warmup

        if 'mm_sdm' in self.current_task:
            # ============ Missing-aware Robust Encoding Integration ============
            if self.use_missing_aware and use_modality_dropout:
                # Step 1: Get full-modal embeddings (for consistency loss)
                full_cone_embeds, full_ctwo_embeds, full_cthree_embeds, full_cfour_embeds = \
                    self.router_multimodal_embeds_with_missing_aware(rgbs, nirs, cps, sks, texts, 
                                                                      modality_mask=[True, True, True, True, True])
                
                # Step 2: Apply modality dropout and get subset embeddings
                modality_mask = self.sample_modality_dropout()
                subset_cone_embeds, subset_ctwo_embeds, subset_cthree_embeds, subset_cfour_embeds = \
                    self.router_multimodal_embeds_with_missing_aware(rgbs, nirs, cps, sks, texts,
                                                                      modality_mask=modality_mask)
                
                # Use the subset embeddings for main training
                cone_embeds = subset_cone_embeds
                ctwo_embeds = subset_ctwo_embeds
                cthree_embeds = subset_cthree_embeds
                cfour_embeds = subset_cfour_embeds
                
                # Store full-modal embeddings for consistency loss
                ret.update({'modality_mask': modality_mask})
            else:
                # Standard processing without modality dropout
                if self.use_missing_aware:
                    cone_embeds, ctwo_embeds, cthree_embeds, cfour_embeds = \
                        self.router_multimodal_embeds_with_missing_aware(rgbs, nirs, cps, sks, texts)
                else:
                    cone_embeds, ctwo_embeds, cthree_embeds, cfour_embeds = \
                        self.router_multimodal_embeds(rgbs, nirs, cps, sks, texts)

            cone_feats = cone_embeds
            ctwo_feats,cthree_feats,cfour_feats = [],[],[]

            for i in range(len(ctwo_embeds)):
                ctwo_embed = ctwo_embeds[i]
                ctwo_feat = self.mm_fusion(ctwo_embed,ctwo_embed,ctwo_embed)
                ctwo_feats.append(ctwo_feat)

            for i in range(len(cthree_embeds)):
                cthree_embed = cthree_embeds[i]
                cthree_feat = self.mm_fusion(cthree_embed, cthree_embed, cthree_embed)
                cthree_feats.append(cthree_feat)

            for i in range(len(cfour_embeds)):
                cfour_embed = cfour_embeds[i]
                cfour_feat = self.mm_fusion(cfour_embed, cfour_embed, cfour_embed)
                cfour_feats.append(cfour_feat)

            rgb_feat = cone_feats[0]
            cone_feats = cone_feats[1:]

            cone_losses,ctwo_losses,cthree_losses,cfour_losses = [],[],[],[]

            for cone_feat in cone_feats:
                cone_losses.append(objectives.compute_sdm(rgb_feat, cone_feat, batch['pids'], logit_scale))

            for ctwo_feat in ctwo_feats:
                ctwo_losses.append(objectives.compute_sdm(rgb_feat, ctwo_feat, batch['pids'], logit_scale))

            for cthree_feat in cthree_feats:
                cthree_losses.append(objectives.compute_sdm(rgb_feat, cthree_feat, batch['pids'], logit_scale))

            for cfour_feat in cfour_feats:
                cfour_losses.append(objectives.compute_sdm(rgb_feat, cfour_feat, batch['pids'], logit_scale))

            cone_aver_loss = torch.mean(torch.stack(cone_losses))
            ctwo_aver_loss = torch.mean(torch.stack(ctwo_losses))
            cthree_aver_loss = torch.mean(torch.stack(cthree_losses))
            cfour_aver_loss = torch.mean(torch.stack(cfour_losses))

            ret.update({'cone_mmsdm_loss': cone_aver_loss})
            ret.update({'ctwo_mmsdm_loss': ctwo_aver_loss})
            ret.update({'cthree_mmsdm_loss': cthree_aver_loss})
            ret.update({'cfour_mmsdm_loss': cfour_aver_loss})
            
            # ============ Consistency Loss for Missing-aware Encoding ============
            if self.use_missing_aware and use_modality_dropout and self.training:
                # Compute full-modal features for consistency loss
                full_cone_feats = full_cone_embeds
                full_ctwo_feats, full_cthree_feats, full_cfour_feats = [], [], []
                
                for i in range(len(full_ctwo_embeds)):
                    full_ctwo_embed = full_ctwo_embeds[i]
                    full_ctwo_feat = self.mm_fusion(full_ctwo_embed, full_ctwo_embed, full_ctwo_embed)
                    full_ctwo_feats.append(full_ctwo_feat)
                
                for i in range(len(full_cthree_embeds)):
                    full_cthree_embed = full_cthree_embeds[i]
                    full_cthree_feat = self.mm_fusion(full_cthree_embed, full_cthree_embed, full_cthree_embed)
                    full_cthree_feats.append(full_cthree_feat)
                
                for i in range(len(full_cfour_embeds)):
                    full_cfour_embed = full_cfour_embeds[i]
                    full_cfour_feat = self.mm_fusion(full_cfour_embed, full_cfour_embed, full_cfour_embed)
                    full_cfour_feats.append(full_cfour_feat)
                
                # Full RGB feature
                full_rgb_feat = full_cone_feats[0]
                
                # Compute consistency loss: align subset features with full-modal features
                consistency_losses = []
                
                # Consistency for RGB
                consistency_losses.append(
                    self.consistency_constraint(full_rgb_feat, rgb_feat)
                )
                
                # Consistency for single modalities
                full_cone_feats_rest = full_cone_feats[1:]
                for full_feat, subset_feat in zip(full_cone_feats_rest, cone_feats):
                    consistency_losses.append(
                        self.consistency_constraint(full_feat, subset_feat)
                    )
                
                # Consistency for two-modal combinations
                for full_feat, subset_feat in zip(full_ctwo_feats, ctwo_feats):
                    consistency_losses.append(
                        self.consistency_constraint(full_feat, subset_feat)
                    )
                
                # Consistency for three-modal combinations
                for full_feat, subset_feat in zip(full_cthree_feats, cthree_feats):
                    consistency_losses.append(
                        self.consistency_constraint(full_feat, subset_feat)
                    )
                
                # Consistency for four-modal combinations
                for full_feat, subset_feat in zip(full_cfour_feats, cfour_feats):
                    consistency_losses.append(
                        self.consistency_constraint(full_feat, subset_feat)
                    )
                
                consistency_loss = torch.mean(torch.stack(consistency_losses))
                ret.update({'consistency_loss': consistency_loss * self.consistency_loss_weight})
            
            # ============ Cross-modal Feature Completion Loss ============
            if self.use_cross_modal_completion and self.training:
                # Extract CLS features for completion training
                # Use cone_embeds which contains [rgb, nir, cp, sk, text_eot] CLS features
                modality_cls_features = {
                    'RGB': cone_embeds[0] if len(cone_embeds) > 0 else rgb_feat,
                    'NIR': cone_feats[0] if len(cone_feats) > 0 else None,
                    'CP': cone_feats[1] if len(cone_feats) > 1 else None,
                    'SK': cone_feats[2] if len(cone_feats) > 2 else None,
                    'TEXT': cone_feats[3] if len(cone_feats) > 3 else None
                }
                # Filter out None values
                modality_cls_features = {k: v for k, v in modality_cls_features.items() if v is not None}
                
                # Only compute completion losses if we have enough modalities
                if len(modality_cls_features) >= 2:
                    # Compute completion losses (reconstruction + cycle consistency)
                    completion_losses = self.compute_completion_losses(
                        modality_cls_features, 
                        compute_cycle=True
                    )
                    ret.update(completion_losses)
            
            # ============ Reliability-Adaptive Fusion Loss ============
            if self.use_reliability_fusion and self.training:
                # Get single modality CLS features for reliability fusion training
                # Use the features from the main computation path
                reliability_modality_features = {
                    'RGB': rgb_feat,
                    'NIR': cone_feats[0] if len(cone_feats) > 0 else None,
                    'CP': cone_feats[1] if len(cone_feats) > 1 else None,
                    'SK': cone_feats[2] if len(cone_feats) > 2 else None,
                    'TEXT': cone_feats[3] if len(cone_feats) > 3 else None
                }
                # Filter out None values
                reliability_modality_features = {k: v for k, v in reliability_modality_features.items() if v is not None}
                
                # Mark all real features as not generated
                is_generated = {name: False for name in reliability_modality_features.keys()}
                
                # Only compute fusion losses if we have enough modalities
                if len(reliability_modality_features) >= 2:
                    _, fusion_losses = self.compute_reliability_fusion(
                        modality_features=reliability_modality_features,
                        is_generated=is_generated,
                        return_losses=True
                    )
                    # Losses are already named 'fusion_sparsity_loss' and 'fusion_uncertainty_loss'
                    ret.update(fusion_losses)

            if 'id' in self.current_task:
                all_feats = [rgb_feat] + cone_feats + ctwo_feats + cthree_feats + cfour_feats
                assert len(all_feats) == 16
                logits_list = []
                for feat in all_feats:
                    logits = self.classifier(feat.half()).float()
                    logits_list.append(logits)
                ret.update({'id_loss': objectives.compute_id(logits_list,batch['pids']) * self.args.id_loss_weight})
                return ret

        if 'mm_itc' in self.current_task:
            # Missing-aware encoding for mm_itc
            if self.use_missing_aware:
                cone_embeds, ctwo_embeds, cthree_embeds, cfour_embeds = \
                    self.router_multimodal_embeds_with_missing_aware(rgbs, nirs, cps, sks, texts)
            else:
                cone_embeds, ctwo_embeds, cthree_embeds, cfour_embeds = \
                    self.router_multimodal_embeds(rgbs, nirs, cps, sks, texts)

            cone_feats = cone_embeds
            ctwo_feats,cthree_feats,cfour_feats = [],[],[]

            for i in range(len(ctwo_embeds)):
                ctwo_embed = ctwo_embeds[i]
                ctwo_feat = self.mm_fusion(ctwo_embed,ctwo_embed,ctwo_embed)
                ctwo_feats.append(ctwo_feat)

            for i in range(len(cthree_embeds)):
                cthree_embed = cthree_embeds[i]
                cthree_feat = self.mm_fusion(cthree_embed, cthree_embed, cthree_embed)
                cthree_feats.append(cthree_feat)

            for i in range(len(cfour_embeds)):
                cfour_embed = cfour_embeds[i]
                cfour_feat = self.mm_fusion(cfour_embed, cfour_embed, cfour_embed)
                cfour_feats.append(cfour_feat)

            rgb_feat = cone_feats[0]
            cone_feats = cone_feats[1:]

            cone_losses,ctwo_losses,cthree_losses,cfour_losses = [],[],[],[]

            for cone_feat in cone_feats:
                cone_losses.append(objectives.compute_itc(rgb_feat, cone_feat, logit_scale))

            for ctwo_feat in ctwo_feats:
                ctwo_losses.append(objectives.compute_itc(rgb_feat, ctwo_feat, logit_scale))

            for cthree_feat in cthree_feats:
                cthree_losses.append(objectives.compute_itc(rgb_feat, cthree_feat, logit_scale))

            for cfour_feat in cfour_feats:
                cfour_losses.append(objectives.compute_itc(rgb_feat, cfour_feat, logit_scale))

            cone_aver_loss = torch.mean(torch.stack(cone_losses))
            ctwo_aver_loss = torch.mean(torch.stack(ctwo_losses))
            cthree_aver_loss = torch.mean(torch.stack(cthree_losses))
            cfour_aver_loss = torch.mean(torch.stack(cfour_losses))

            ret.update({'cone_mmitc_loss': cone_aver_loss})
            ret.update({'ctwo_mmitc_loss': ctwo_aver_loss})
            ret.update({'cthree_mmitc_loss': cthree_aver_loss})
            ret.update({'cfour_mmitc_loss': cfour_aver_loss})
            return ret

        if 'mm_supitc' in self.current_task:
            # Missing-aware encoding for mm_supitc
            if self.use_missing_aware:
                cone_embeds, ctwo_embeds, cthree_embeds, cfour_embeds = \
                    self.router_multimodal_embeds_with_missing_aware(rgbs, nirs, cps, sks, texts)
            else:
                cone_embeds, ctwo_embeds, cthree_embeds, cfour_embeds = \
                    self.router_multimodal_embeds(rgbs, nirs, cps, sks, texts)

            cone_feats = cone_embeds
            ctwo_feats,cthree_feats,cfour_feats = [],[],[]

            for i in range(len(ctwo_embeds)):
                ctwo_embed = ctwo_embeds[i]
                ctwo_feat = self.mm_fusion(ctwo_embed,ctwo_embed,ctwo_embed)
                ctwo_feats.append(ctwo_feat)

            for i in range(len(cthree_embeds)):
                cthree_embed = cthree_embeds[i]
                cthree_feat = self.mm_fusion(cthree_embed, cthree_embed, cthree_embed)
                cthree_feats.append(cthree_feat)

            for i in range(len(cfour_embeds)):
                cfour_embed = cfour_embeds[i]
                cfour_feat = self.mm_fusion(cfour_embed, cfour_embed, cfour_embed)
                cfour_feats.append(cfour_feat)

            rgb_feat = cone_feats[0]
            cone_feats = cone_feats[1:]

            cone_losses,ctwo_losses,cthree_losses,cfour_losses = [],[],[],[]

            for cone_feat in cone_feats:
                cone_losses.append(objectives.compute_supitc(rgb_feat, cone_feat, batch['pids']))

            for ctwo_feat in ctwo_feats:
                ctwo_losses.append(objectives.compute_supitc(rgb_feat, ctwo_feat, batch['pids']))

            for cthree_feat in cthree_feats:
                cthree_losses.append(objectives.compute_supitc(rgb_feat, cthree_feat, batch['pids']))

            for cfour_feat in cfour_feats:
                cfour_losses.append(objectives.compute_supitc(rgb_feat, cfour_feat, batch['pids']))

            cone_aver_loss = torch.mean(torch.stack(cone_losses))
            ctwo_aver_loss = torch.mean(torch.stack(ctwo_losses))
            cthree_aver_loss = torch.mean(torch.stack(cthree_losses))
            cfour_aver_loss = torch.mean(torch.stack(cfour_losses))

            ret.update({'cone_mmsupitc_loss': cone_aver_loss})
            ret.update({'ctwo_mmsupitc_loss': ctwo_aver_loss})
            ret.update({'cthree_mmsupitc_loss': cthree_aver_loss})
            ret.update({'cfour_mmsupitc_loss': cfour_aver_loss})
            return ret

        if 'mm_cmpm' in self.current_task:
            # Missing-aware encoding for mm_cmpm
            if self.use_missing_aware:
                cone_embeds, ctwo_embeds, cthree_embeds, cfour_embeds = \
                    self.router_multimodal_embeds_with_missing_aware(rgbs, nirs, cps, sks, texts)
            else:
                cone_embeds, ctwo_embeds, cthree_embeds, cfour_embeds = \
                    self.router_multimodal_embeds(rgbs, nirs, cps, sks, texts)

            cone_feats = cone_embeds
            ctwo_feats,cthree_feats,cfour_feats = [],[],[]

            for i in range(len(ctwo_embeds)):
                ctwo_embed = ctwo_embeds[i]
                ctwo_feat = self.mm_fusion(ctwo_embed,ctwo_embed,ctwo_embed)
                ctwo_feats.append(ctwo_feat)

            for i in range(len(cthree_embeds)):
                cthree_embed = cthree_embeds[i]
                cthree_feat = self.mm_fusion(cthree_embed, cthree_embed, cthree_embed)
                cthree_feats.append(cthree_feat)

            for i in range(len(cfour_embeds)):
                cfour_embed = cfour_embeds[i]
                cfour_feat = self.mm_fusion(cfour_embed, cfour_embed, cfour_embed)
                cfour_feats.append(cfour_feat)

            rgb_feat = cone_feats[0]
            cone_feats = cone_feats[1:]

            cone_losses,ctwo_losses,cthree_losses,cfour_losses = [],[],[],[]

            for cone_feat in cone_feats:
                cone_losses.append(objectives.compute_cmpm(rgb_feat, cone_feat, batch['pids']))

            for ctwo_feat in ctwo_feats:
                ctwo_losses.append(objectives.compute_cmpm(rgb_feat, ctwo_feat, batch['pids']))

            for cthree_feat in cthree_feats:
                cthree_losses.append(objectives.compute_cmpm(rgb_feat, cthree_feat, batch['pids']))

            for cfour_feat in cfour_feats:
                cfour_losses.append(objectives.compute_cmpm(rgb_feat, cfour_feat, batch['pids']))

            cone_aver_loss = torch.mean(torch.stack(cone_losses))
            ctwo_aver_loss = torch.mean(torch.stack(ctwo_losses))
            cthree_aver_loss = torch.mean(torch.stack(cthree_losses))
            cfour_aver_loss = torch.mean(torch.stack(cfour_losses))

            ret.update({'cone_mmcmpm_loss': cone_aver_loss})
            ret.update({'ctwo_mmcmpm_loss': ctwo_aver_loss})
            ret.update({'cthree_mmcmpm_loss': cthree_aver_loss})
            ret.update({'cfour_mmcmpm_loss': cfour_aver_loss})
            return ret

        rgb_feats = self.encode_rgb_cls(rgbs)
        nir_feats = self.encode_nir_cls(nirs)
        cp_feats = self.encode_cp_cls(cps)
        sk_feats = self.encode_sk_cls(sks)
        text_feats = self.encode_text_cls(texts)

        if 'itc' in self.current_task:
            ret.update({'nir_itc_loss': objectives.compute_itc(rgb_feats, nir_feats, logit_scale)})
            ret.update({'cp_itc_loss': objectives.compute_itc(rgb_feats, cp_feats, logit_scale)})
            ret.update({'sk_itc_loss': objectives.compute_itc(rgb_feats, sk_feats, logit_scale)})
            ret.update({'txt_itc_loss': objectives.compute_itc(rgb_feats, text_feats, logit_scale)})

        if 'sdm' in self.current_task:
            ret.update({'nir_sdm_loss': objectives.compute_sdm(rgb_feats, nir_feats, batch['pids'], logit_scale)})
            ret.update({'cp_sdm_loss': objectives.compute_sdm(rgb_feats, cp_feats, batch['pids'], logit_scale)})
            ret.update({'sk_sdm_loss': objectives.compute_sdm(rgb_feats, sk_feats, batch['pids'], logit_scale)})
            ret.update({'txt_sdm_loss': objectives.compute_sdm(rgb_feats, text_feats, batch['pids'], logit_scale)})
        return ret

def build_model(args, num_classes=11003):
    model = ReID5oModel(args, num_classes)
    # covert model to fp16
    convert_weights(model)
    return model