import torch

def make_optimizer(cfg, model):

    lr = cfg.SOLVER.BASE_LR
    weight_decay = cfg.SOLVER.WEIGHT_DECAY
    momentum = cfg.SOLVER.MOMENTUM

    params_to_update = []
    params_names = []
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            params_names.append(name)

    optimizer = torch.optim.SGD(params_to_update, lr=lr,
                weight_decay=weight_decay, momentum=momentum)
    # optimizer = torch.optim.AdamW(
    #     params_to_update,
    #     lr=lr,
    #     weight_decay=weight_decay
    # )

    return optimizer


def make_lr_scheduler(cfg, optimizer):
    # T_max = cfg.SOLVER.MAX_EPOCH

    # eta_min = cfg.SOLVER.BASE_LR


    # return torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer,
    #     T_max=T_max,  
    #     eta_min=eta_min  
    # )
    step_size = cfg.SOLVER.STEPS
    gamma = cfg.SOLVER.GAMMA
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)