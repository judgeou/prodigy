import math
from typing import TYPE_CHECKING, Any, Callable, Optional

import torch
import torch.optim
import logging
import os
import torch.distributed as dist

if TYPE_CHECKING:
    from torch.optim.optimizer import _params_t
else:
    _params_t = Any


class Prodigy(torch.optim.Optimizer):
    r"""
    Implements Adam with Prodigy step-sizes.
    Leave LR set to 1 unless you encounter instability.
   
    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float):
            Learning rate adjustment parameter. Increases or decreases the Prodigy learning rate.
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        beta3 (float):
            coefficients for computing the Prodidy stepsize using running averages.
            If set to None, uses the value of square root of beta2 (default: None).
        eps (float):
            Term added to the denominator outside of the root operation to improve numerical stability. (default: 1e-8).
        weight_decay (float):
            Weight decay, i.e. a L2 penalty (default: 0).
        decouple (boolean):
            Use AdamW style decoupled weight decay
        use_bias_correction (boolean):
            Turn on Adam's bias correction. Off by default.
        safeguard_warmup (boolean):
            Remove lr from the denominator of D estimate to avoid issues during warm-up stage. Off by default.
        d0 (float):
            Initial D estimate for D-adaptation (default 1e-6). Rarely needs changing.
        d_coef (float):
            Coefficient in the expression for the estimate of d (default 1.0).
            Values such as 0.5 and 2.0 typically work as well. 
            Changing this parameter is the preferred way to tune the method.
        growth_rate (float):
            prevent the D estimate from growing faster than this multiplicative rate.
            Default is inf, for unrestricted. Values like 1.02 give a kind of learning
            rate warmup effect.
        fsdp_in_use (bool):
            If you're using sharded parameters, this should be set to True. The optimizer
            will attempt to auto-detect this, but if you're using an implementation other
            than PyTorch's builtin version, the auto-detection won't work.
        slice_p (int): Reduce memory usage by calculating LR adaptation statistics on only every 
            pth entry of each tensor. For values greater than 1 this is an approximation to standard 
            Prodigy. Values ~11 are reasonable (default 1).
        cosine_decay (bool): Enable cosine decay after reaching peak learning rate (default False).
        peak_patience (int): Number of consecutive steps with no d growth to consider peak reached (default 50).
        total_training_steps (int): Total number of training steps. When peak is detected, cosine decay will 
            use remaining steps automatically. If None, defaults to 1000 steps from peak (default None).
    """
    def __init__(self, params, lr=1.0,
                 betas=(0.9, 0.999), beta3=None,
                 eps=1e-8, weight_decay=0, decouple=True, 
                 use_bias_correction=False, safeguard_warmup=False,
                 d0=1e-6, d_coef=1.0, growth_rate=float('inf'),
                 fsdp_in_use=False,
                 slice_p=1,
                 cosine_decay=False, peak_patience=50, total_training_steps=None):
        if not 0.0 < d0:
            raise ValueError("Invalid d0 value: {}".format(d0))
        if not 0.0 < lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 < eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        if decouple and weight_decay > 0:
            print(f"Using decoupled weight decay")

       
        defaults = dict(lr=lr, betas=betas, beta3=beta3,
                        eps=eps, weight_decay=weight_decay,
                        d=d0, d0=d0, d_max=d0,
                        d_numerator=0.0, d_coef=d_coef,
                        k=0, growth_rate=growth_rate,
                        use_bias_correction=use_bias_correction,
                        decouple=decouple, safeguard_warmup=safeguard_warmup,
                        fsdp_in_use=fsdp_in_use,
                        slice_p=slice_p,
                        cosine_decay=cosine_decay, peak_patience=peak_patience,
                        total_training_steps=total_training_steps,
                        peak_step=0, no_growth_count=0, in_decay_phase=False,
                        cosine_decay_steps=1000)
        self.d0 = d0
        super().__init__(params, defaults)

    @property
    def supports_memory_efficient_fp16(self):
        return False

    @property
    def supports_flat_params(self):
        return True

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        d_denom = 0.0

        group = self.param_groups[0]
        use_bias_correction = group['use_bias_correction']
        beta1, beta2 = group['betas']
        beta3 = group['beta3']
        if beta3 is None:
            beta3 = math.sqrt(beta2)
        k = group['k']

        d = group['d']
        d_max = group['d_max']
        d_coef = group['d_coef']
        lr = max(group['lr'] for group in self.param_groups)

        if use_bias_correction:
            bias_correction = ((1 - beta2**(k+1))**0.5) / (1 - beta1**(k+1))
        else:
            bias_correction = 1

        dlr = d*lr*bias_correction
       
        growth_rate = group['growth_rate']
        decouple = group['decouple']
        fsdp_in_use = group['fsdp_in_use']

        d_numerator = group['d_numerator']
        d_numerator *= beta3
        delta_numerator = 0.0

        for group in self.param_groups:
            decay = group['weight_decay']
            k = group['k']
            eps = group['eps']
            group_lr = group['lr']
            d0 = group['d0']
            safeguard_warmup = group['safeguard_warmup']
            slice_p = group['slice_p']

            if group_lr not in [lr, 0.0]:
                raise RuntimeError(f"Setting different lr values in different parameter groups is only supported for values of 0")

            for p in group['params']:
                if p.grad is None:
                    continue
                if hasattr(p, "_fsdp_flattened"):
                    fsdp_in_use = True
               
                grad = p.grad.data
               
                # Apply weight decay (coupled variant)
                if decay != 0 and not decouple:
                    grad.add_(p.data, alpha=decay)

                state = self.state[p]

                # State initialization
                if 'step' not in state:
                    state['step'] = 0

                    state['s'] = torch.zeros_like(p.data.flatten()[::slice_p]).detach()

                    if p.any():
                        state['p0'] = p.flatten()[::slice_p].detach().clone()
                    else:
                        # All values are zero, so save VRAM with a zero-tensor
                        state['p0'] = torch.tensor(0, device=p.device, dtype=p.dtype)

                    # Exponential moving average of gradient values
                    if beta1 > 0:
                        state['exp_avg'] = torch.zeros_like(p.data).detach()
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data).detach()

                exp_avg_sq = state['exp_avg_sq']

                s = state['s']
                p0 = state['p0']

                if group_lr > 0.0:
                    # we use d / d0 instead of just d to avoid getting values that are too small
                    sliced_grad = grad.flatten()[::slice_p]
                    delta_numerator += (d / d0) * dlr * torch.dot(sliced_grad, p0.data - p.data.flatten()[::slice_p]).item()

                    # Adam EMA updates
                    if beta1 > 0:
                        exp_avg = state['exp_avg']
                        exp_avg.mul_(beta1).add_(grad, alpha=d * (1-beta1))
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=d * d * (1-beta2))

                    if safeguard_warmup:
                        s.mul_(beta3).add_(sliced_grad, alpha=((d / d0) * d))
                    else:
                        s.mul_(beta3).add_(sliced_grad, alpha=((d / d0) * dlr))
                    d_denom += s.abs().sum().item()

            ######

        d_hat = d

        # if we have not done any progres, return
        # if we have any gradients available, will have d_denom > 0 (unless \|g\|=0)
        if d_denom == 0 and not fsdp_in_use:
            return loss
       
        if lr > 0.0:
            if fsdp_in_use:
                dist_tensor = torch.zeros(2).cuda()
                dist_tensor[0] = delta_numerator
                dist_tensor[1] = d_denom
                dist.all_reduce(dist_tensor, op=dist.ReduceOp.SUM)
                global_d_numerator = d_numerator + dist_tensor[0]
                global_d_denom = dist_tensor[1]
            else:
                global_d_numerator = d_numerator + delta_numerator
                global_d_denom = d_denom

            d_hat = d_coef * global_d_numerator / global_d_denom
            if d == group['d0']:
                d = max(d, d_hat)
            
            # Check for cosine decay logic first
            cosine_decay_enabled = group['cosine_decay']
            in_decay_phase = group['in_decay_phase'] if cosine_decay_enabled else False
            
            # Store previous d_max to detect growth
            prev_d_max = d_max
            
            # Only update d_max if not in decay phase
            if not in_decay_phase:
                d_max = max(d_max, d_hat)
            
            if cosine_decay_enabled:
                peak_patience = group['peak_patience']
                total_training_steps = group['total_training_steps']
                no_growth_count = group['no_growth_count']
                in_decay_phase = group['in_decay_phase']
                peak_step = group['peak_step']
                
                # Check if d_max has grown
                if d_max > prev_d_max:
                    # Reset counter when we see growth
                    no_growth_count = 0
                else:
                    # Increment counter when no growth
                    no_growth_count += 1
                
                # Enter decay phase if we've had no growth for peak_patience steps
                if not in_decay_phase and no_growth_count >= peak_patience:
                    in_decay_phase = True
                    peak_step = k
                    
                    # Calculate remaining steps for cosine decay
                    if total_training_steps is not None:
                        remaining_steps = max(1, total_training_steps - k)
                        print(f"Peak detected at step {k}, starting cosine decay over {remaining_steps} remaining steps")
                    else:
                        remaining_steps = 1000  # fallback default
                        print(f"Peak detected at step {k}, starting cosine decay over {remaining_steps} steps (default)")
                    
                    # Store the cosine decay steps in the group
                    group['cosine_decay_steps'] = remaining_steps
                
                # Apply cosine decay if in decay phase
                if in_decay_phase:
                    cosine_decay_steps = group['cosine_decay_steps']
                    decay_progress = min((k - peak_step) / cosine_decay_steps, 1.0)
                    cosine_factor = 0.5 * (1 + math.cos(math.pi * decay_progress))
                    d_proposed = d_max * cosine_factor
                    
                    # Check if the natural d estimate suggests we should go higher
                    d_natural = min(d_max, d * growth_rate)
                    
                    # If natural estimate is significantly higher, reset cosine state
                    if d_natural > d_proposed * 1.05:  # 5% threshold to avoid noise
                        print(f"Step {k}: Learning rate wants to rise (natural: {d_natural:.6f} vs cosine: {d_proposed:.6f}), resetting cosine decay")
                        in_decay_phase = False
                        no_growth_count = 0
                        peak_step = 0
                        d = d_natural
                    else:
                        d = d_proposed
                else:
                    d = min(d_max, d * growth_rate)
                
                # Update group state
                group['no_growth_count'] = no_growth_count
                group['in_decay_phase'] = in_decay_phase
                group['peak_step'] = peak_step
            else:
                d = min(d_max, d * growth_rate)

        for group in self.param_groups:
            group['d_numerator'] = global_d_numerator
            group['d_denom'] = global_d_denom
            group['d'] = d
            group['d_max'] = d_max
            group['d_hat'] = d_hat

            decay = group['weight_decay']
            k = group['k']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]

                exp_avg_sq = state['exp_avg_sq']

                state['step'] += 1

                denom = exp_avg_sq.sqrt().add_(d * eps)

                # Apply weight decay (decoupled variant)
                if decay != 0 and decouple:
                    p.data.add_(p.data, alpha=-decay * dlr)

                ### Take step
                if beta1 > 0:
                    exp_avg = state['exp_avg']
                    p.data.addcdiv_(exp_avg, denom, value=-dlr)
                else:
                    p.data.addcdiv_(grad, denom, value=-dlr * d)

            group['k'] = k + 1

        return loss
        
