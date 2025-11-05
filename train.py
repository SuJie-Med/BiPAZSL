import os
from os.path import join
import argparse
import torch
import numpy as np
import random
from data import build_dataloader
from model.modeling import build_gzsl_pipeline
from model.solver import make_optimizer, make_lr_scheduler
from model.engine.trainer import do_train
from model.config import cfg
from model.utils.comm import synchronize, is_main_process
from model.utils import ReDirectSTD

# Import mixed precision training module (APEX)
try:
    from apex import amp
except ImportError:
    raise ImportError('Please install APEX for mixed-precision training: https://github.com/NVIDIA/apex')


def train_model(cfg, local_rank, distributed, resume):
    """
    Main training function for zero-shot learning models.
    
    Args:
        cfg: Configuration object with training parameters
        local_rank: Local GPU rank for distributed training
        distributed: Whether using distributed training
        resume: Path to checkpoint for resuming training
    
    Returns:
        Trained model
    """
    # Set random seeds for reproducibility
    seed = 3407
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Build model architecture
    model = build_gzsl_pipeline(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model = model.to(device)

    # Initialize optimizer and learning rate scheduler
    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)

    # Configure mixed precision training
    use_mixed_precision = cfg.DTYPE == "float16"
    amp_opt_level = 'O1' if use_mixed_precision else 'O0'
    model, optimizer = amp.initialize(model, optimizer, opt_level=amp_opt_level)

    # Wrap model for distributed training if needed
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True
        )

    # Build data loaders
    tr_dataloader, tu_loader, ts_loader, res = build_dataloader(
        cfg,
        is_distributed=distributed
    )

    # Prepare model saving path
    output_dir = cfg.OUTPUT_DIR
    model_file_name = cfg.MODEL_FILE_NAME
    model_file_path = join(output_dir, model_file_name)

    # Training parameters from config
    test_gamma = cfg.TEST.GAMMA
    max_epoch = cfg.SOLVER.MAX_EPOCH
    loss_weights = {
        1: cfg.MODEL.LOSS.LAMBDA1,  # Classification loss weight
        2: cfg.MODEL.LOSS.LAMBDA2,  # Regression loss weight
        3: cfg.MODEL.LOSS.LAMBDA3,  # Bias loss weight
        4: cfg.MODEL.LOSS.LAMBDA4   # Contrastive loss weight
    }

    # Resume from checkpoint if specified
    start_epoch = 0
    if resume and os.path.exists(resume):
        print(f"Loading checkpoint from {resume}...")
        checkpoint = torch.load(resume, map_location=device)
        
        # Load model weights
        model.load_state_dict(checkpoint['model'])
        
        # Load optimizer state if available
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        
        # Get starting epoch
        start_epoch = checkpoint.get('epoch', 0)
        print(f"Resuming training from epoch {start_epoch + 1}")

    # Main training loop
    do_train(
        model=model,
        tr_dataloader=tr_dataloader,
        tu_loader=tu_loader,
        ts_loader=ts_loader,
        res=res,
        optimizer=optimizer,
        scheduler=scheduler,
        lamd=loss_weights,
        test_gamma=test_gamma,
        device=device,
        max_epoch=max_epoch,
        model_file_path=model_file_path,
        start_epoch=start_epoch
    )

    return model


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Zero-Shot Learning Model Training")
    parser.add_argument(
        "--config-file",
        default="config/sun.yaml",
        metavar="FILE",
        help="Path to configuration file",
        type=str,
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=0,
        help="Local GPU rank for distributed training"
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        help="Path to checkpoint for resuming training (default: start from scratch)"
    )
    args = parser.parse_args()

    # Initialize distributed training settings
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl",
            init_method="env://"
        )
        synchronize()

    # Load and freeze configuration
    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    # Setup logging (redirect stdout to log file for main process)
    output_dir = cfg.OUTPUT_DIR
    log_file_name = cfg.LOG_FILE_NAME
    log_file_path = join(output_dir, log_file_name)

    if is_main_process():
        ReDirectSTD(log_file_path, 'stdout', True)

    # Print configuration information
    print(f"Loaded configuration file: {args.config_file}")
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        print(config_str)
    print(f"Running with config:\n{cfg}")

    # Enable cuDNN benchmark for faster training
    torch.backends.cudnn.benchmark = True
    # Enable anomaly detection for debugging (disable in production)
    torch.autograd.set_detect_anomaly(True)

    # Start training
    train_model(cfg, args.local_rank, args.distributed, args.resume)


if __name__ == '__main__':
    # Set visible GPUs (modify according to available devices)
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
    main()
