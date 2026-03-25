import os
import math

import torch
from torch.optim import lr_scheduler

from .TSONet import TSONet


def create_model(opt):
    if opt.model == 'tsonet':
        model = TSONet(
            opt=opt,
            dim=opt.dim,
            hidden_dim=getattr(opt, 'hidden_dim', 256),
            n_bins=getattr(opt, 'num_height_bins', 64),
            mem_levels=tuple(getattr(opt, "mem_levels", [5, 4, 3])),
        )
    else:
        raise NotImplementedError('Model [%s] is not recognized' % opt.model)

    model = model.to(opt.device)
    print('Using {} model\n'.format(opt.model))

    return model


def resume_check(model, optimizer, scheduler, opt):
    checkpoint_path = os.path.join(opt.save_dir, '{}.pth'.format(opt.ckpt_name))

    if os.path.exists(checkpoint_path) and opt.resume:
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=opt.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_epoch = checkpoint['best_epoch']
        best_metric = checkpoint.get('best_metric', checkpoint.get('best_value', float('inf') if opt.mode in ['reg', 'multi'] else float('-inf')))
        print(f"Checkpoint loaded, resuming training from epoch {start_epoch}\n")
        print(f"Previous best epoch: {best_epoch}, best metric: {best_metric:.4f}")
    else:
        print("No checkpoint found, starting from scratch")
        start_epoch = 1
        best_epoch = 1
        if opt.mode in ['reg', 'multi']:
            best_metric = float('inf')
        else:
            best_metric = float('-inf')

    return model, optimizer, scheduler, start_epoch, best_metric, best_epoch


def create_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_freq, gamma=opt.lr_gamma)

    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)

    elif opt.lr_policy == 'warmcos':
        total_epochs = opt.n_epochs
        warmup_epochs = max(1, int(total_epochs * 0.3))
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return float(epoch + 1) / float(warmup_epochs)   # linear warmup to 1.0
            t = (epoch - warmup_epochs) / max(1, (total_epochs - warmup_epochs))
            return 0.5 * (1.0 + math.cos(math.pi * t))
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    elif opt.lr_policy == 'plateau':
        if opt.mode == 'seg':
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, threshold=1e-4, patience=5)
        else:
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, threshold=1e-4, patience=5)
    # elif opt.lr_policy == 'cycle':
    #     scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, epochs=E, steps_per_epoch=steps_per_epoch,
    #                                         pct_start=0.3, div_factor=25, final_div_factor=1e3)
    else:
        raise NotImplementedError('Learning rate policy [%s] is not implemented' % opt.lr_policy)

    return scheduler
