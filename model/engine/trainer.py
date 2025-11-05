import torch
import numpy as np

import torch.distributed as dist
from models.utils.comm import *
from .inferencer import eval_zs_gzsl
from apex import amp

def find_best_gamma(model, tu_loader, ts_loader, res, device, gamma_range=np.arange(0.1, 5.1, 0.2)):
    """
    Search for the optimal gamma value on the validation set to maximize H-score.
    Args:
        model: Trained model to evaluate
        tu_loader: DataLoader for test-unseen dataset
        ts_loader: DataLoader for test-seen dataset
        res: Dictionary containing dataset information (attributes, labels, etc.)
        device: Computing device (CPU/GPU)
        gamma_range: Range of candidate gamma values (default: 0.1 to 5.0 with step 0.2)
    Returns:
        best_gamma: Gamma value with the highest H-score
        best_H: Corresponding highest H-score
    """
    model.eval()  # Switch to evaluation mode
    best_H = -1.0
    best_gamma = 0.0
    with torch.no_grad():  # Disable gradient computation for speed
        for gamma in gamma_range:
            # Evaluate with current gamma
            acc_seen, acc_unseen, H, acc_zs = eval_zs_gzsl(
                tu_loader, ts_loader, res, model, gamma, device
            )
            # Update best gamma if current H-score is higher
            if H > best_H:
                best_H = H
                best_gamma = gamma
    model.train()  # Switch back to training mode
    return best_gamma, best_H

def reduce_loss_dict(loss_dict):
    """
    Reduce loss dictionaries from all processes so that the process with rank 0
    contains the averaged results. Returns a dictionary with the same keys as
    the input, after reduction.
    Args:
        loss_dict: Dictionary of loss values from a single process
    Returns:
        reduced_losses: Dictionary of averaged loss values (only valid for rank 0)
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        # Collect all loss names and values
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        # Stack losses and reduce across processes
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)  # Sum losses from all processes
        if dist.get_rank() == 0:
            all_losses /= world_size  # Average losses for rank 0
        # Reconstruct reduced loss dictionary
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses

def do_train(
        model,
        tr_dataloader,
        tu_loader,
        ts_loader,
        res,
        optimizer,
        scheduler,
        lamd,
        test_gamma,
        device,
        max_epoch,
        model_file_path,
        start_epoch=0
    ):
    """
    Main training function for the model. Handles training loops, loss computation,
    validation, and model saving based on performance.
    Args:
        model: Model to be trained
        tr_dataloader: Training data loader
        tu_loader: Test-unseen data loader
        ts_loader: Test-seen data loader
        res: Dictionary with dataset metadata (attributes, labels, etc.)
        optimizer: Optimizer for parameter updates
        scheduler: Learning rate scheduler
        lamd: List of loss weights [_, cls_weight, reg_weight, bias_weight, con_weight]
        test_gamma: Default gamma for non-main processes during validation
        device: Computing device (CPU/GPU)
        max_epoch: Total number of training epochs
        model_file_path: Path to save the best model checkpoint
        start_epoch: Starting epoch (for resuming training, default: 0)
    """
    # Initialize best performance tracking [ZSL_acc, seen_acc, unseen_acc, H-score]
    best_performance = [0, 0, 0, 0]
    best_epoch = -1

    # Prepare attribute tensors for the model (move to device)
    att_seen = res['att_seen'].to(device)
    att_unseen = res['att_unseen'].to(device)
    att = torch.cat((att_seen, att_unseen), dim=0)  # Combine seen and unseen attributes

    # Lists to track loss trends across epochs
    losses = []
    cls_losses = []
    reg_losses = []
    bias_losses = []
    scale_all = []

    model.train()  # Set model to training mode

    # Main training loop
    for epoch in range(start_epoch, max_epoch):
        # Track losses within the current epoch
        loss_epoch = []
        cls_loss_epoch = []
        reg_loss_epoch = []
        con_loss_epoch = []
        bias_loss_epoch = []
        scale_epoch = []

        # Iterate over training batches
        for iteration, (batch_img, batch_att, batch_label) in enumerate(tr_dataloader):
            # Move batch data to device
            batch_img = batch_img.to(device)
            batch_att = batch_att.to(device)
            batch_label = batch_label.to(device)
            
            # Forward pass: compute losses
            loss_dict = model(
                x=batch_img, 
                att=batch_att, 
                label=batch_label, 
                seen_att=att_seen,
                att_all=att
            )

            # Extract individual loss components
            Lreg = loss_dict['Reg_loss']
            Lcls = loss_dict['Cls_loss']
            Lbias = loss_dict['bias_loss']
            Lcon = loss_dict['con_loss']
            scale = loss_dict['scale']
            loss_dict.pop('scale')  # Remove scale from loss dict (not a loss)
            

            # Total loss with weighted components
            loss = lamd[1]*Lcls + lamd[2]*Lreg + lamd[3]*Lbias + lamd[4]*Lcon
            
            # Reduce losses across processes (for distributed training)
            loss_dict_reduced = reduce_loss_dict(loss_dict)

            # Extract reduced loss components
            lreg = loss_dict_reduced['Reg_loss']
            lcls = loss_dict_reduced['Cls_loss']
            lbias = loss_dict_reduced['bias_loss']
            lcon = loss_dict_reduced['con_loss']
            losses_reduced = lamd[1]*lcls + lamd[2]*lreg + lamd[3]*lbias + lamd[4]*lcon
            
            # Backward pass and optimization
            optimizer.zero_grad()  # Clear gradients

            # Mixed precision training (via apex)
            with amp.scale_loss(loss, optimizer) as scaled_losses:
                scaled_losses.backward()
            optimizer.step()  # Update parameters

            # Track losses for the current epoch
            loss_epoch.append(losses_reduced.item())
            cls_loss_epoch.append(lcls.item())
            reg_loss_epoch.append(lreg.item())
            bias_loss_epoch.append(lbias.item())
            con_loss_epoch.append(lcon.item())
            scale_epoch.append(scale)

        # Update learning rate scheduler
        scheduler.step()

        # Log training metrics (only for main process)
        if is_main_process():
            losses += loss_epoch
            scale_all += scale_epoch
            # Compute epoch-averaged metrics
            loss_epoch_mean = sum(loss_epoch)/len(loss_epoch)
            scale_epoch_mean = sum(scale_epoch) / len(scale_epoch)
            con_loss_epoch_mean = sum(con_loss_epoch)/len(con_loss_epoch)
            losses_mean = sum(losses) / len(losses)
            scale_all_mean = sum(scale_all) / len(scale_all)

            # Format and print training log
            log_info = 'epoch: %d  |  loss: %.4f,  cls: %.4f,  reg: %.4f,  bias: %.4f,  con: %.4f,  scale: %.4f,  lr: %.6f' % \
                       (epoch + 1, loss_epoch_mean, lcls.item(), lreg.item(), lbias.item(), con_loss_epoch_mean, scale_epoch_mean, optimizer.param_groups[0]["lr"])
            print(log_info)

        # Synchronize all processes before validation
        synchronize()

        # Search for optimal gamma (only main process performs search)
        if is_main_process():
            best_gamma, _ = find_best_gamma(
                model, tu_loader, ts_loader, res, device,
                gamma_range=np.arange(0.1, 5.1, 0.2)  # Step 0.2 balances efficiency and precision
            )
            print(f"Epoch {epoch+1} optimal gamma: {best_gamma:.2f}")
        else:
            best_gamma = test_gamma  # Use default gamma for non-main processes

        # Synchronize optimal gamma across all processes (required for distributed training)
        # best_gamma = broadcast_tensor(torch.tensor(best_gamma, device=device), 0).item()

        # Evaluate zero-shot and generalized zero-shot performance
        acc_seen, acc_novel, H, acc_zs = eval_zs_gzsl(
            tu_loader,
            ts_loader,
            res,
            model,
            best_gamma,
            device)

        # Synchronize after evaluation
        synchronize()

        # Log evaluation results and update best model (main process only)
        if is_main_process():
            print(f'zsl: {acc_zs:.4f}, gzsl: seen={acc_seen:.4f}, unseen={acc_novel:.4f}, h={H:.4f}')
            
            # Update best ZSL performance (if current is better)
            if acc_zs > best_performance[0]:
                best_performance[0] = acc_zs
         
            # Update best GZSL performance (based on H-score) and save model
            if H > best_performance[-1]:
                best_epoch = epoch + 1
                best_performance = [acc_zs, acc_seen, acc_novel, H]
                
                # Save model checkpoint
                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "best_performance": best_performance
                }
                torch.save(checkpoint, model_file_path)
                print(f'save model (H best): {model_file_path}')

    # Print final best performance after training completes (main process)
    if is_main_process():
        print(f"Best GZSL epoch: {best_epoch}")
        print(f'zsl: {best_performance[0]:.4f}, gzsl: seen={best_performance[1]:.4f}, unseen={best_performance[2]:.4f}, h={best_performance[3]:.4f}')
        print(f"Current best ZSL accuracy: {best_performance[0]:.4f}")
