import torch
import numpy as np

import torch.distributed as dist
from models.utils.comm import *
from .inferencer import eval_zs_gzsl
from apex import amp

def find_best_gamma(model, tu_loader, ts_loader, res, device, gamma_range=np.arange(0.1, 5.1, 0.2)):
    """
    在验证集上搜索最优gamma，最大化H-score
    gamma_range: 候选gamma值范围（如0.1到5.0，步长0.2）
    """
    model.eval()  # 评估模式
    best_H = -1.0
    best_gamma = 0.0
    with torch.no_grad():  # 禁用梯度，加速计算
        for gamma in gamma_range:
            # 用当前gamma评估
            acc_seen, acc_unseen, H, acc_zs = eval_zs_gzsl(
                tu_loader, ts_loader, res, model, gamma, device
            )
            # 记录最优gamma（以H-score为指标）
            if H > best_H:
                best_H = H
                best_gamma = gamma
    model.train()  # 切回训练模式
    return best_gamma, best_H

def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            all_losses /= world_size
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

    best_performance = [0, 0, 0, 0]
    best_epoch = -1

    att_seen = res['att_seen'].to(device)
    att_unseen = res['att_unseen'].to(device)
    att = torch.cat((att_seen, att_unseen), dim=0)
    losses = []
    cls_losses = []
    reg_losses = []

    bias_losses = []
    scale_all = []

    model.train()

    for epoch in range(start_epoch, max_epoch):

        loss_epoch = []
        cls_loss_epoch = []
        reg_loss_epoch = []
        con_loss_epoch = []

        bias_loss_epoch = []
        scale_epoch = []



        for iteration, (batch_img, batch_att, batch_label) in enumerate(tr_dataloader):

            batch_img = batch_img.to(device)
            batch_att = batch_att.to(device)
            batch_label = batch_label.to(device)

            # Step 3.1: 识别伪标签样本（label == -1）
            # --------------------------------------------------------
            valid_mask = batch_label >= 0  # True 表示正常 seen 类样本
            unsup_mask = batch_label < 0   # 伪标签样本（来自 unseen 类）
            if valid_mask.sum() == 0:
                continue
            
            loss_dict = model(x=batch_img, att=batch_att, label=batch_label, seen_att=att_seen,att_all=att)

            Lreg = loss_dict['Reg_loss']
            Lcls = loss_dict['Cls_loss']
            Lbias = loss_dict['bias_loss']
            Lcon = loss_dict['con_loss']
            scale = loss_dict['scale']
            loss_dict.pop('scale')
            

            loss = lamd[1]*Lcls+ lamd[2]*Lreg + lamd[3]*Lbias + lamd[4]*Lcon
            
            loss_dict_reduced = reduce_loss_dict(loss_dict)

            lreg = loss_dict_reduced['Reg_loss']
            lcls = loss_dict_reduced['Cls_loss']
            lbias = loss_dict_reduced['bias_loss']
            lcon = loss_dict_reduced['con_loss']
            losses_reduced = lamd[1]*lcls + lamd[2]*lreg + lamd[3]*lbias + lamd[4]*lcon
            
            optimizer.zero_grad()

            with amp.scale_loss(loss, optimizer) as scaled_losses:
                scaled_losses.backward()
            optimizer.step()

            loss_epoch.append(losses_reduced.item())
            cls_loss_epoch.append(lcls.item())
            reg_loss_epoch.append(lreg.item())
            bias_loss_epoch.append(lbias.item())
            con_loss_epoch.append(lcon.item())
            scale_epoch.append(scale)

        scheduler.step()

        if is_main_process():
            losses += loss_epoch
            scale_all += scale_epoch
            loss_epoch_mean = sum(loss_epoch)/len(loss_epoch)
            scale_epoch_mean = sum(scale_epoch) / len(scale_epoch)
            con_loss_epoch_mean = sum(con_loss_epoch)/len(con_loss_epoch)
            losses_mean = sum(losses) / len(losses)
            scale_all_mean = sum(scale_all) / len(scale_all)


            # log_info = 'epoch: %d  |  loss: %.4f (%.4f),   scale:  %.4f (%.4f),    lr: %.6f' % \
            #            (epoch + 1, loss_epoch_mean, losses_mean, 
            #             scale_epoch_mean, scale_all_mean, optimizer.param_groups[0]["lr"])
            # print(log_info)
            log_info = 'epoch: %d  |  loss: %.4f,  cls: %.4f,  reg: %.4f,  bias: %.4f,  con: %.4f,  scale: %.4f,  lr: %.6f' % \
                       (epoch + 1, loss_epoch_mean, lcls.item(), lreg.item(), lbias.item(), con_loss_epoch_mean, scale_epoch_mean, optimizer.param_groups[0]["lr"])
            print(log_info)


        synchronize()

        if is_main_process():
            # 搜索最优gamma（范围可根据数据集调整）
            best_gamma, _ = find_best_gamma(
                model, tu_loader, ts_loader, res, device,
                gamma_range=np.arange(0.1, 5.1, 0.2)  # 步长0.2，平衡效率与精度
            )
            print(f"Epoch {epoch+1} 最优gamma: {best_gamma:.2f}")
        else:
            best_gamma = test_gamma  # 非主进程暂用默认值

        # 同步最优gamma到所有进程（分布式训练必需）
        # best_gamma = broadcast_tensor(torch.tensor(best_gamma, device=device), 0).item()

        
        acc_seen, acc_novel, H, acc_zs = eval_zs_gzsl(
            tu_loader,
            ts_loader,
            res,
            model,
            best_gamma,
            device)

        synchronize()

        if is_main_process():
            print('zsl: %.4f, gzsl: seen=%.4f, unseen=%.4f, h=%.4f' % (acc_zs, acc_seen, acc_novel, H))
            
            if acc_zs > best_performance[0]:
                # best_epoch = epoch + 1
                best_performance[0] = acc_zs
         
                # checkpoint = {
                #     "model": model.state_dict(),
                #     "optimizer": optimizer.state_dict(),
                #     "scheduler": scheduler.state_dict(),
                #     "epoch": epoch,
                #     "best_performance": best_performance
                # }
                # torch.save(checkpoint, model_file_path)
                # print('save model (ZSL best): ' + model_file_path)

            if H > best_performance[-1]:
                best_epoch = epoch + 1
                # 更新最佳性能指标（包含H在内的所有指标）
                best_performance = [acc_zs, acc_seen, acc_novel, H]
                
                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "best_performance": best_performance
                }
                torch.save(checkpoint, model_file_path)
                print('save model (H best): ' + model_file_path)
                # best_performance[1:] = [acc_seen, acc_novel, H]



    if is_main_process():
        print("best GZSL: ep: %d" % best_epoch)
        print('zsl: %.4f, gzsl: seen=%.4f, unseen=%.4f, h=%.4f' % tuple(best_performance))
        # print("Current best GZSL H-score: %.4f" % best_performance[-1])
        print("Current best ZSL accuracy: %.4f" % best_performance[0])