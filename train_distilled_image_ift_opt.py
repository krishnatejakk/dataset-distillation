import logging
import time
import copy
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from basics import task_loss, final_objective_loss, evaluate_steps
from utils.distributed import broadcast_coalesced, all_reduce_coalesced
from utils.io import save_results


def permute_list(input_list):
    indices = np.random.permutation(len(input_list))
    return [input_list[i] for i in indices]


class Trainer(object):
    def __init__(self, state, models):
        self.state = state
        self.models = models
        self.num_data_steps = state.distill_steps  # how much data we have
        self.T = state.distill_steps * state.distill_epochs  # how many sc steps we run
        self.num_per_step = state.num_classes * state.distilled_images_per_class_per_step
        assert state.distill_lr >= 0, 'distill_lr must >= 0'
        self.init_data_optim()

    def init_data_optim(self):
        self.params = []
        state = self.state
        optim_lr = state.lr

        # labels
        self.labels = []
        distill_label = torch.arange(state.num_classes, dtype=torch.long, device=state.device) \
                             .repeat(state.distilled_images_per_class_per_step, 1)  # [[0, 1, 2, ...], [0, 1, 2, ...]]
        distill_label = distill_label.t().reshape(-1)  # [0, 0, ..., 1, 1, ...]
        for _ in range(self.num_data_steps):
            self.labels.append(distill_label)
        self.all_labels = torch.cat(self.labels)

        # data
        self.data = []
        for _ in range(self.num_data_steps):
            distill_data = torch.randn(self.num_per_step, state.nc, state.input_size, state.input_size,
                                       device=state.device, requires_grad=True)
            self.data.append(distill_data)
            self.params.append(distill_data)

        assert len(self.params) > 0, "must have at least 1 parameter"

        # now all the params are in self.params, sync if using distributed
        if state.distributed:
            broadcast_coalesced(self.params)
            logging.info("parameters broadcast done!")

        self.optimizer = optim.SGD(self.params, lr=optim_lr, momentum=0.9, weight_decay=5e-4)
        # self.optimizer = optim.AdamW(self.params, lr=optim_lr, betas=(0.5, 0.999), weight_decay=5e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=state.decay_epochs,
                                                   gamma=state.decay_factor)
        for p in self.params:
            p.grad = torch.zeros_like(p)

    def get_steps(self):
        data_label_iterable = (x for _ in range(self.state.distill_epochs) for x in zip(self.data, self.labels))
        steps = []
        for (data, label) in data_label_iterable:
            steps.append((data, label))
        return steps

    def save_results(self, steps=None, visualize=True, subfolder=''):
        with torch.no_grad():
            steps = steps or self.get_steps()
            save_results(self.state, steps, visualize=visualize, subfolder=subfolder)

    def __call__(self):
        return self.train()

    def prefetch_train_loader_iter(self):
        state = self.state
        device = state.device
        train_iter = iter(state.train_loader)
        for epoch in range(state.epochs):
            niter = len(train_iter)
            prefetch_it = max(0, niter - 2)
            for it, val in enumerate(train_iter):
                # Prefetch (start workers) at the end of epoch BEFORE yielding
                if it == prefetch_it and epoch < state.epochs - 1:
                    train_iter = iter(state.train_loader)
                yield epoch, it, val

    def train(self):
        state = self.state
        device = state.device
        train_loader = state.train_loader
        sample_n_nets = state.local_sample_n_nets
        grad_divisor = state.sample_n_nets  # i.e., global sample_n_nets
        ckpt_int = state.checkpoint_interval

        data_t0 = time.time()

        for epoch, it, (rdata, rlabel) in self.prefetch_train_loader_iter():
            data_t = time.time() - data_t0

            if it == 0:
                self.scheduler.step()

            if it == 0 and ((ckpt_int >= 0 and epoch % ckpt_int == 0) or epoch == 0):
                with torch.no_grad():
                    steps = self.get_steps()
                self.save_results(steps=steps, subfolder='checkpoints/epoch{:04d}'.format(epoch))
                evaluate_steps(state, steps, 'Begin of epoch {}'.format(epoch))

            do_log_this_iter = it == 0 or (state.log_interval >= 0 and it % state.log_interval == 0)

            self.optimizer.zero_grad()
            rdata, rlabel = rdata.to(device, non_blocking=True), rlabel.to(device, non_blocking=True)

            if sample_n_nets == state.local_n_nets:
                tmodels = self.models
            else:
                idxs = np.random.choice(state.local_n_nets, sample_n_nets, replace=False)
                tmodels = [self.models[i] for i in idxs]

            t0 = time.time()
            losses = []
            steps = self.get_steps()

            # activate everything needed to run on this process
            grad_infos = []
            for model in tmodels:
                if state.train_nets_type == 'unknown_init':
                    model.reset(state)
                model.train()
                
                # params = list(model.parameters())
                # grouped_params = [list(p) for p in zip(*params)]
                
                for data, label in steps:
                    output = model(data)
                    loss = task_loss(state, output, label)
                    loss.backward()
            
                    with torch.no_grad():
                        for param in model.parameters():
                            param -= state.distill_lr * param.grad
                            param.grad.zero_()
            
                # Final Training Loss
                loss = 0
                for i, (data, label) in enumerate(zip(self.data, self.labels)):
                    output = model(data)
                    loss += task_loss(state, output, label) / len(self.data)
                
                # Final Validation Loss
                val_loss = final_objective_loss(state, model(rdata), rlabel)
                
                # Compute Hyper-Gradient using Implicit Differentiation
                v = torch.autograd.grad(val_loss, model.parameters(), retain_graph=True)
                f = torch.autograd.grad(loss, model.parameters(), create_graph=True)
                
                # Compute Approx. Inverse Hessian Vector Product
                p = [v_i.detach().clone() for v_i in v]
                
                for _ in range(20):
                    old_v = list(v)
                    with torch.enable_grad():
                        temp1 = torch.autograd.grad(f, model.parameters(), retain_graph=True, grad_outputs=v)
                    temp1 = list(temp1)
                    v = list(v)
                    for k in range(len(v)):
                        v[k] = old_v[k] - (state.distill_lr * temp1[k])
                    v = tuple(old_v)
                    for k in range(len(v)):
                        p[k] += v[k]
                p = tuple([-x for x in p])
                
                # Compute gradients
                with torch.enable_grad():
                    v3 = torch.autograd.grad(f, self.params, grad_outputs=p)
                
                # Store gradients
                grad_infos.append(v3)
                self.optimizer.zero_grad()
                for p in self.params:
                    p.grad = torch.zeros_like(p)
                
                for grads in grad_infos:
                    for g, p in zip(grads, self.params):
                        p.grad.data.add_(g.data)
                
                losses.append(val_loss.detach())
                
                # all reduce if needed
                all_reduce_tensors = [p.grad for p in model.parameters()]
                if do_log_this_iter:
                    losses = torch.stack(losses, 0).sum()
                    all_reduce_tensors.append(losses)
                
                if state.distributed:
                    torch.distributed.all_reduce_coalesced(all_reduce_tensors, grad_divisor)
                else:
                    for t in all_reduce_tensors:
                        t.div_(grad_divisor)
                
                # Step the optimizer
                self.optimizer.step()
                self.optimizer.zero_grad()
                
            t1 = time.time()
            t = t1 - t0

            if do_log_this_iter:
                loss = losses.item()
                logging.info((
                    'Epoch: {:4d} [{:7d}/{:7d} ({:2.0f}%)]\tLoss: {:.4f}\t'
                    'Data Time: {:.2f}s\tTrain Time: {:.2f}s'
                ).format(
                    epoch, it * train_loader.batch_size, len(train_loader.dataset),
                    100. * it / len(train_loader), loss, data_t, t,
                ))
                if loss != loss:  # nan
                    raise RuntimeError('loss became NaN')

            del steps, grad_infos, losses, all_reduce_tensors

            data_t0 = time.time()

        # End of Training
        with torch.no_grad():
            steps = self.get_steps()
        self.save_results(steps=steps, subfolder='final')
        evaluate_steps(state, steps, 'End of Training')

        return steps
    
def distill(state, models):
    return Trainer(state, models).train()
