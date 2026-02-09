import os
import os.path as op
import time
import logging
import torch
import torch.nn as nn

# Optional TensorBoard support
# try:
#     from torch.utils.tensorboard import SummaryWriter
#     TENSORBOARD_AVAILABLE = True
# except ImportError:
#     TENSORBOARD_AVAILABLE = False
#     print("Warning: TensorBoard not available. Install with: pip install tensorboard")

from torch.utils.tensorboard import SummaryWriter

from datasets import build_dataloader
from model import build_model
from solver import build_optimizer, build_lr_scheduler
from utils.checkpoint import Checkpointer
from utils.logger import setup_logger
from utils.meter import AverageMeter
from utils.metrics import Evaluator
from utils.options import get_args
from utils.iotools import save_train_configs, mkdir_if_missing
from utils.comm import get_rank, synchronize

TENSORBOARD_AVAILABLE = True

def train_one_epoch(args, model, train_loader, optimizer, scheduler, epoch, checkpointer, tensorboard_writer=None):
    """
    Train for one epoch
    
    Supports Missing-aware Robust Encoding when args.use_missing_aware is True:
    - Applies modality dropout during training
    - Computes consistency loss between full-modal and subset embeddings
    """
    logger = logging.getLogger("ORBench.train")
    device = next(model.parameters()).device
    
    model.train()
    
    # Prepare loss meters
    loss_meters = {}
    batch_time = AverageMeter()
    data_time = AverageMeter()
    
    # Track modality dropout statistics if enabled
    use_missing_aware = getattr(args, 'use_missing_aware', False)
    if use_missing_aware:
        modality_presence_counts = {'RGB': 0, 'NIR': 0, 'CP': 0, 'SK': 0, 'TEXT': 0}
        total_batches = 0
    
    end = time.time()
    
    for iteration, batch in enumerate(train_loader):
        data_time.update(time.time() - end)
        
        # Move data to device
        for key in batch:
            if torch.is_tensor(batch[key]):
                batch[key] = batch[key].to(device)
        
        # Forward pass (modality dropout is handled inside the model)
        # Pass current epoch for warmup control
        ret = model(batch, current_epoch=epoch)
        
        # Track modality presence if using missing-aware encoding
        if use_missing_aware and 'modality_mask' in ret:
            modality_mask = ret['modality_mask']
            modality_names = ['RGB', 'NIR', 'CP', 'SK', 'TEXT']
            for i, present in enumerate(modality_mask):
                if present:
                    modality_presence_counts[modality_names[i]] += 1
            total_batches += 1
        
        # Aggregate losses with warmup handling
        total_loss = 0
        loss_dict = {}
        
        # Get warmup epoch settings
        modality_dropout_warmup = getattr(args, 'modality_dropout_warmup_epochs', 5)
        completion_warmup = getattr(args, 'completion_warmup_epochs', 3)
        fusion_warmup = getattr(args, 'fusion_warmup_epochs', 5)
        
        for loss_name, loss_value in ret.items():
            if 'loss' in loss_name:
                # Apply warmup: gradually introduce auxiliary losses
                loss_weight = 1.0
                
                # Consistency loss warmup (tied to modality dropout warmup)
                if 'consistency' in loss_name:
                    if epoch <= modality_dropout_warmup:
                        # Gradually ramp up: 0 at epoch 1, full at warmup+1
                        loss_weight = max(0, (epoch - 1) / modality_dropout_warmup)
                
                # Completion losses warmup
                if 'completion' in loss_name:
                    if epoch <= completion_warmup:
                        loss_weight = max(0, (epoch - 1) / completion_warmup)
                
                # Fusion regularization losses warmup
                if 'fusion_sparsity' in loss_name or 'fusion_uncertainty' in loss_name:
                    if epoch <= fusion_warmup:
                        loss_weight = max(0, (epoch - 1) / fusion_warmup)
                
                weighted_loss = loss_value * loss_weight
                total_loss += weighted_loss
                loss_dict[loss_name] = loss_value.item()
                
                # Initialize loss meter if not exists
                if loss_name not in loss_meters:
                    loss_meters[loss_name] = AverageMeter()
                loss_meters[loss_name].update(loss_value.item(), args.batch_size)
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        batch_time.update(time.time() - end)
        end = time.time()
        
        # Logging
        if (iteration + 1) % args.log_period == 0:
            log_str = f"Epoch[{epoch}] Iteration[{iteration + 1}/{len(train_loader)}] "
            log_str += f"Time {batch_time.val:.3f} ({batch_time.avg:.3f}) "
            log_str += f"Data {data_time.val:.3f} ({data_time.avg:.3f}) "
            log_str += f"LR {scheduler.get_last_lr()[0]:.2e} "
            
            loss_str = " ".join([f"{k}: {v.avg:.4f}" for k, v in loss_meters.items()])
            log_str += loss_str
            
            logger.info(log_str)
            
            # TensorBoard logging
            if tensorboard_writer is not None and get_rank() == 0:
                global_step = epoch * len(train_loader) + iteration
                tensorboard_writer.add_scalar('Train/Learning_Rate', scheduler.get_last_lr()[0], global_step)
                for loss_name, meter in loss_meters.items():
                    tensorboard_writer.add_scalar(f'Train/{loss_name}', meter.val, global_step)
    
    # Log modality dropout statistics at end of epoch
    if use_missing_aware and total_batches > 0:
        logger.info("Modality Presence Statistics (Missing-aware Encoding):")
        for modality, count in modality_presence_counts.items():
            presence_rate = count / total_batches * 100
            logger.info(f"  {modality}: {presence_rate:.1f}% present")
    
    return loss_meters


def validate(args, model, val_loaders, epoch):
    """
    Validate the model
    """
    logger = logging.getLogger("ORBench.validate")
    logger.info(f"Validation at Epoch {epoch}")
    
    # Create evaluator with all validation loaders
    # get_mAP controls whether to compute mAP/mINP (slower but more comprehensive)
    compute_mAP = getattr(args, 'compute_mAP', False)
    evaluator = Evaluator(
        gallery_loader=val_loaders[0],
        get_mAP=compute_mAP,
        # Single modality
        nir_query_loader=val_loaders[1],
        cp_query_loader=val_loaders[2],
        sk_query_loader=val_loaders[3],
        text_query_loader=val_loaders[4],
        # Two modalities
        nir_cp_query_loader=val_loaders[5],
        cp_nir_query_loader=val_loaders[6],
        nir_sk_query_loader=val_loaders[7],
        sk_nir_query_loader=val_loaders[8],
        nir_text_query_loader=val_loaders[9],
        text_nir_query_loader=val_loaders[10],
        cp_sk_query_loader=val_loaders[11],
        sk_cp_query_loader=val_loaders[12],
        cp_text_query_loader=val_loaders[13],
        text_cp_query_loader=val_loaders[14],
        sk_text_query_loader=val_loaders[15],
        text_sk_query_loader=val_loaders[16],
        # Three modalities
        nir_cp_sk_query_loader=val_loaders[17],
        cp_nir_sk_query_loader=val_loaders[18],
        sk_nir_cp_query_loader=val_loaders[19],
        nir_cp_text_query_loader=val_loaders[20],
        cp_nir_text_query_loader=val_loaders[21],
        text_nir_cp_query_loader=val_loaders[22],
        nir_sk_text_query_loader=val_loaders[23],
        sk_nir_text_query_loader=val_loaders[24],
        text_nir_sk_query_loader=val_loaders[25],
        cp_sk_text_query_loader=val_loaders[26],
        sk_cp_text_query_loader=val_loaders[27],
        text_cp_sk_query_loader=val_loaders[28],
        # Four modalities
        nir_cp_sk_text_query_loader=val_loaders[29],
        cp_nir_sk_text_query_loader=val_loaders[30],
        sk_nir_cp_text_query_loader=val_loaders[31],
        text_nir_cp_sk_query_loader=val_loaders[32]
    )
    
    top1 = evaluator.eval(model.eval())
    
    return top1


def main():
    args = get_args()
    
    # Create output directory
    mkdir_if_missing(args.output_dir)
    
    # Setup logger
    logger = setup_logger('ORBench', save_dir=args.output_dir, if_train=args.training)
    logger.info(args)
    
    # Save config
    save_train_configs(args.output_dir, args)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Build dataloaders
    logger.info("Building dataloaders...")
    data_loaders = build_dataloader(args)
    
    # Extract train and validation loaders
    train_loader = data_loaders[0]
    val_loaders = data_loaders[1:-1]  # All loaders except train_loader and num_classes
    num_classes = data_loaders[-1]
    
    logger.info(f"Number of training batches: {len(train_loader)}")
    logger.info(f"Number of classes: {num_classes}")
    
    # Build model
    logger.info("Building model...")
    model = build_model(args, num_classes=num_classes)
    model.to(device)
    
    # Log Missing-aware Robust Encoding configuration
    if getattr(args, 'use_missing_aware', False):
        logger.info("=" * 50)
        logger.info("Missing-aware Robust Encoding ENABLED")
        logger.info(f"  Modality dropout prob: {getattr(args, 'modality_dropout_prob', 0.5)}")
        logger.info(f"  Keep RGB prob: {getattr(args, 'keep_rgb_prob', 0.8)}")
        logger.info(f"  Min modalities keep: {getattr(args, 'min_modalities_keep', 1)}")
        logger.info(f"  Consistency loss weight: {getattr(args, 'consistency_loss_weight', 0.5)}")
        logger.info(f"  Consistency loss type: {getattr(args, 'consistency_loss_type', 'cosine')}")
        logger.info("=" * 50)
    
    # Log Cross-modal Feature Completion configuration
    if getattr(args, 'use_cross_modal_completion', False):
        logger.info("=" * 50)
        logger.info("Cross-modal Feature Completion ENABLED")
        logger.info(f"  Num heads: {getattr(args, 'completion_num_heads', 8)}")
        logger.info(f"  Num layers: {getattr(args, 'completion_num_layers', 2)}")
        logger.info(f"  Dropout: {getattr(args, 'completion_dropout', 0.1)}")
        logger.info(f"  Reconstruction loss weight: {getattr(args, 'completion_recon_loss_weight', 1.0)}")
        logger.info(f"  Cycle loss weight: {getattr(args, 'completion_cycle_loss_weight', 0.5)}")
        logger.info(f"  Loss type: {getattr(args, 'completion_loss_type', 'cosine')}")
        logger.info(f"  Use during inference: {getattr(args, 'use_completion_inference', True)}")
        logger.info("=" * 50)
    
    # Log Reliability-Adaptive Fusion configuration
    if getattr(args, 'use_reliability_fusion', False):
        logger.info("=" * 50)
        logger.info("Reliability-Adaptive Fusion ENABLED")
        logger.info(f"  Hidden dim: {getattr(args, 'reliability_hidden_dim', 256)}")
        logger.info(f"  Num heads: {getattr(args, 'reliability_num_heads', 8)}")
        logger.info(f"  Num layers: {getattr(args, 'reliability_num_layers', 2)}")
        logger.info(f"  Use quality indicators: {getattr(args, 'use_quality_indicators', True)}")
        logger.info(f"  Use transformer refinement: {getattr(args, 'use_transformer_refinement', True)}")
        logger.info(f"  Sparsity weight: {getattr(args, 'fusion_sparsity_weight', 0.1)}")
        logger.info(f"  Uncertainty weight: {getattr(args, 'fusion_uncertainty_weight', 0.2)}")
        logger.info(f"  Sparsity target: {getattr(args, 'fusion_sparsity_target', 0.3)}")
        logger.info(f"  Sparsity type: {getattr(args, 'fusion_sparsity_type', 'entropy')}")
        logger.info(f"  Use during inference: {getattr(args, 'use_reliability_fusion_inference', True)}")
        logger.info("=" * 50)
    
    # Build optimizer
    logger.info("Building optimizer...")
    optimizer = build_optimizer(args, model)
    
    # Build scheduler
    logger.info("Building learning rate scheduler...")
    scheduler = build_lr_scheduler(args, optimizer)
    
    # Setup checkpointer
    checkpointer = Checkpointer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        save_dir=args.output_dir,
        save_to_disk=True,
        logger=logger
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_top1 = 0.0
    
    if args.resume:
        if args.resume_ckpt_file:
            checkpoint = checkpointer.resume(f=args.resume_ckpt_file)
            start_epoch = checkpoint.get('epoch', 0)
            best_top1 = checkpoint.get('best_top1', 0.0)
            logger.info(f"Resumed from epoch {start_epoch}, best top1: {best_top1:.4f}")
        else:
            logger.warning("Resume flag is set but no checkpoint file provided!")
    
    # Setup TensorBoard
    tensorboard_writer = None
    if TENSORBOARD_AVAILABLE and get_rank() == 0:
        tensorboard_dir = op.join(args.output_dir, 'tensorboard')
        mkdir_if_missing(tensorboard_dir)
        tensorboard_writer = SummaryWriter(tensorboard_dir)
        logger.info(f"TensorBoard directory: {tensorboard_dir}")
    
    # Training loop
    logger.info("Start training")
    for epoch in range(start_epoch, args.num_epoch):
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch + 1}/{args.num_epoch}")
        logger.info(f"{'='*50}")
        
        # Random sampling of modality pairings at the start of each epoch
        logger.info("Random sampling modality pairings...")
        train_loader.dataset.random_sampling()
        
        # Train one epoch
        loss_meters = train_one_epoch(
            args, model, train_loader, optimizer, 
            scheduler, epoch + 1, checkpointer, tensorboard_writer
        )
        
        # Step the scheduler
        scheduler.step()
        
        # Validation
        if (epoch + 1) % args.eval_period == 0:
            top1 = validate(args, model, val_loaders, epoch + 1)
            
            # Log validation results
            if tensorboard_writer is not None and get_rank() == 0:
                tensorboard_writer.add_scalar('Val/Top1_Avg', top1, epoch + 1)
            
            # Save checkpoint
            is_best = top1 > best_top1
            if is_best:
                best_top1 = top1
                logger.info(f"New best Top1: {best_top1:.4f}")
            
            if get_rank() == 0:
                checkpointer.save(
                    name=f"epoch_{epoch + 1}",
                    epoch=epoch + 1,
                    best_top1=best_top1
                )
                
                if is_best:
                    checkpointer.save(
                        name="best",
                        epoch=epoch + 1,
                        best_top1=best_top1
                    )
        
        # Save periodic checkpoint
        if get_rank() == 0 and (epoch + 1) % 10 == 0:
            checkpointer.save(
                name=f"checkpoint_epoch_{epoch + 1}",
                epoch=epoch + 1,
                best_top1=best_top1
            )
        
        synchronize()
    
    # Save final model
    if get_rank() == 0:
        checkpointer.save(
            name="final",
            epoch=args.num_epoch,
            best_top1=best_top1
        )
    
    logger.info(f"\nTraining completed! Best Top1: {best_top1:.4f}")
    
    if tensorboard_writer is not None:
        tensorboard_writer.close()


if __name__ == '__main__':
    main()
