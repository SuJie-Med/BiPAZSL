import os
from os.path import join
import argparse
import torch
import numpy as np
import random
from data import build_dataloader
from model.modeling import build_gzsl_pipeline
from model.config import cfg
from model.utils.comm import synchronize, is_main_process
from model.utils import ReDirectSTD
from model.engine.inferencer import eval_zs_gzsl


def test_model(cfg, distributed):
    """
    Load trained model and evaluate on test sets (seen/unseen classes).
    
    Args:
        cfg: Configuration object containing test parameters
        distributed: Whether using distributed evaluation
    
    Returns:
        Evaluated model
    """
    # Build the model architecture
    model = build_gzsl_pipeline(cfg) 
    model_dict = model.state_dict()
    
    # Load trained weights (filter unmatched keys)
    saved_dict = torch.load('checkpoints/aaa_sun9.pth')
    saved_dict = {k: v for k, v in saved_dict['model'].items() if k in model_dict}
    model_dict.update(saved_dict)
    model.load_state_dict(model_dict)

    # Move model to target device
    device = torch.device(cfg.MODEL.DEVICE)
    model = model.to(device)

    # Build dataloaders for test sets
    _, tu_loader, ts_loader, res = build_dataloader(
        cfg, 
        is_distributed=distributed
    )

    # Evaluate using specified gamma parameter
    test_gamma = cfg.TEST.GAMMA
    acc_seen, acc_novel, H, acc_zs = eval_zs_gzsl(
        tu_loader,    # Test-unseen dataloader
        ts_loader,    # Test-seen dataloader
        res,          # Dataset metadata
        model,        # Trained model
        test_gamma,   # Gamma parameter for score calibration
        device        # Computing device
    )

    # Print evaluation results
    print(f'ZSL Acc: {acc_zs:.4f}, GZSL: Seen={acc_seen:.4f}, Unseen={acc_novel:.4f}, H-score={H:.4f}')

    return model


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Zero-Shot Learning Model Evaluation")
    parser.add_argument(
        "--config-file",
        default="config/sun.yaml",
        metavar="FILE",
        help="Path to configuration file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed evaluation")
    args = parser.parse_args()

    # Initialize distributed settings
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

    # Print configuration info
    print(f"Loaded configuration file: {args.config_file}")
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        print(config_str)
    print(f"Running with config:\n{cfg}")
        
    # Enable cuDNN benchmark for faster inference
    torch.backends.cudnn.benchmark = True

    # Run model evaluation
    test_model(cfg, args.distributed)


if __name__ == '__main__':
    # Set visible GPUs (modify according to available devices)
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
    main()
