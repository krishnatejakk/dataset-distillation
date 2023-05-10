import logging
import time
import math
import copy
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch_optimizer as th_optim
from basics import task_loss, final_objective_loss, evaluate_steps
from utils.distributed import broadcast_coalesced, all_reduce_coalesced
from utils.io import save_results
import datasets
from collections import defaultdict


def permute_list(list):
    indices = np.random.permutation(len(list))
    return [list[i] for i in indices]


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
        train_dataset = datasets.get_dataset(state, 'train')

        # Initialize the labels_to_idx dictionary
        labels_to_idx = defaultdict(list)
        for idx, (_, label) in enumerate(train_dataset):
            labels_to_idx[label].append(idx)

        for step in range(self.num_data_steps):
            distill_data = torch.randn(self.num_per_step, state.nc, state.input_size, state.input_size,
                                       device=state.device, requires_grad=True)
            distill_label = self.labels[step]
            if state.init_image == 'real':
                #Assign distll_data to real image data corresponding to the labels
                for idx, label in enumerate(distill_label):
                    # Get the list of indices corresponding to the label
                    idx_list = labels_to_idx[label.item()]
                    # Choose a random index from the list
                    rand_idx = np.random.choice(idx_list)
                    # Get the image corresponding to the index
                    img = train_dataset[rand_idx][0]
                    # Convert the image to a tensor
                    img = img.to(state.device)
                    # Reshape the image
                    img = img.reshape(1, state.nc, state.input_size, state.input_size)
                    # Assign the image to the distill_data tensor while allowing gradients
                    distill_data.data[idx] = img.detach().data
                    # Remove the selected index from the dictionary
                    idx_list.remove(rand_idx)
            #set distil data to have requi
            self.data.append(distill_data)
            self.params.append(distill_data)

        assert len(self.params) > 0, "must have at least 1 parameter"

        # now all the params are in self.params, sync if using distributed
        if state.distributed:
            broadcast_coalesced(self.params)
            logging.info("parameters broadcast done!")

        # self.optimizer = optim.SGD(self.params, 
        #                            lr=optim_lr, 
        #                            momentum=0.9, 
        #                            weight_decay=state.weight_decay, 
        #                            nesterov=True)
        
        # self.optimizer = th_optim.MADGRAD(self.params,
        #                                     lr=optim_lr,
        #                                     momentum=0.9,
        #                                     weight_decay=5e-4,
        #                                     eps=1e-6,)

        self.optimizer = th_optim.Ranger(self.params,
                                        lr=optim_lr,
                                        alpha=0.5,
                                        k=6,
                                        N_sma_threshhold=5,
                                        betas=(.95, 0.999),
                                        eps=1e-5,
                                        weight_decay=state.weight_decay,)
        
        # self.optimizer = th_optim.DiffGrad(self.params,
        #                                     lr= optim_lr,
        #                                     betas=(0.9, 0.999),
        #                                     eps=1e-8,
        #                                     weight_decay=5e-4,)
        
        # self.optimizer = optim.Adam(self.params, lr=optim_lr, betas=(0.5, 0.999))
        # self.optimizer = optim.AdamW(self.params, lr=optim_lr, betas=(0.5, 0.999), weight_decay=5e-4)
        # self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=100, T_mult=2, eta_min=0.001)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=state.decay_epochs, gamma=state.decay_factor)
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
        epoch_counter = 0
        for epoch, it, (rdata, rlabel) in self.prefetch_train_loader_iter():
            data_t = time.time() - data_t0

            if (epoch_counter%(2*state.decay_epochs) == 0) and (epoch_counter != 0):
                print("Weight decay is multiplied at Epoch {}".format(epoch))
                self.optimizer.param_groups[0]['weight_decay'] *= state.decay_factor
            
            if it == 0:
                epoch_counter += 1
                self.scheduler.step()

            if it == 0 and ((ckpt_int >= 0 and epoch % ckpt_int == 0) or epoch == 0):
                with torch.no_grad():
                    steps = self.get_steps()
                self.save_results(steps=steps, subfolder='checkpoints/epoch{:04d}'.format(epoch))
                evaluate_steps(state, steps, 'Begin of epoch {}'.format(epoch))

            do_log_this_iter = (it == 0) or (state.log_interval >= 0 and it % state.log_interval == 0)

            self.optimizer.zero_grad()
            # rdata, rlabel = rdata.to(device, non_blocking=True), rlabel.to(device, non_blocking=True)

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
            for net_idx in range(sample_n_nets):
                tmodels = self.models[net_idx : len(self.models): state.local_n_nets]
                sel_model = None
                prev_loss = -math.inf
                for model in tmodels:
                    if state.train_nets_type == 'unknown_init':
                        model.reset(state)
                    model.train()
      
                    for step_i, (data, label) in enumerate(steps):
                        # model_optimizer.zero_grad()
                        output = model(data)
                        loss = task_loss(state, output, label)
                        loss.backward()
                        # model_optimizer.step()
                        with torch.no_grad():
                            for param in model.parameters():
                                param.sub_(state.distill_lr * param.grad)
                                param.grad.zero_()

                    #Final Training Loss
                    loss = 0
                    cnt = len(self.data)
                    for x in zip(self.data, self.labels):
                        output = model(x[0])
                        loss += task_loss(state, output, x[1])/cnt

                    if prev_loss < loss:
                        sel_model = model
                        prev_loss = loss
               
                #Final Training Loss
                loss = 0
                cnt = len(self.data)
                for x in zip(self.data, self.labels):
                    output = sel_model(x[0])
                    loss += task_loss(state, output, x[1])/cnt
                
                #Compute Hyper-Gradient using Implicit Differentiation
                val_loss_item = 0
        
                # Loop over rdata, rlabel in batches of size 1000
                for local_it in range(0, len(rdata), 1024):
                    temp_rdata = rdata[local_it:local_it+1024]
                    temp_rlabel = rlabel[local_it:local_it+1024]
                    temp_rdata, temp_rlabel = temp_rdata.to(device, non_blocking=True), temp_rlabel.to(device, non_blocking=True)
                    #Final Validation Loss
                    val_loss = final_objective_loss(state, sel_model(temp_rdata), temp_rlabel)*(len(temp_rdata)/len(rdata))
                    #Compute Gradient
                    if local_it == 0:
                        v = torch.autograd.grad(val_loss, sel_model.parameters())
                    else:
                        v = tuple([v[i] + torch.autograd.grad(val_loss, sel_model.parameters())[i] for i in range(len(v))])
                    val_loss_item += val_loss.detach()
                    
                f = torch.autograd.grad(loss, sel_model.parameters(), retain_graph=True, create_graph=True)

                #Compute Approx. Inverse Hessian Vector Product
                p  = list(copy.deepcopy(v))
                
                for _ in range(state.neumann_terms_cnt):
                    old_v = list(v)
                    temp1 = torch.autograd.grad(f, sel_model.parameters(), retain_graph=True, grad_outputs=v)
                    temp1 = list(temp1)
                    v = list(v)
                    for k in range(len(v)):
                        v[k] = old_v[k] - (state.distill_lr * temp1[k])
                    v = tuple(old_v)
                    for k in range(len(v)):
                        p[k] += v[k]
                p = tuple(p)
                v3 = torch.autograd.grad(f, self.params, grad_outputs=p)
                param_grads = [-x for x in list(v3)]

                #Update Parameters
                for param, grad in zip(self.params, param_grads):
                    if param.grad is None:
                        param.grad = grad
                    else:
                        param.grad.add_(grad)

                losses.append(val_loss_item)
            
            # all reduce if needed
            # average grad
            all_reduce_tensors = [p.grad for p in self.params]
            if do_log_this_iter:
                losses = torch.stack(losses, 0).sum()
                all_reduce_tensors.append(losses)

            if state.distributed:
                all_reduce_coalesced(all_reduce_tensors, grad_divisor)
            else:
                for t in all_reduce_tensors:
                    t.div_(grad_divisor)

            # Step the optimizer
            self.optimizer.step()
            self.optimizer.zero_grad()
                
            t = time.time() - t0

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

        with torch.no_grad():
            steps = self.get_steps()
        self.save_results(steps)
        return steps


def distill(state, models):
    return Trainer(state, models).train()
