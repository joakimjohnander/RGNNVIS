import glob
import time
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from rgnnvis.utils.stats import AverageMeter
from rgnnvis.utils.debugging import print_tensor_statistics, debug_plot
from rgnnvis.utils.recursive_functions import recursive_detach, recursive_to


class VISTrainer:
    def __init__(self, model, optimizer, data_transform, objective, lr_sched, train_loader, val_loaders,
                 save_name, device,
                 checkpoint_epochs, print_interval, visualization_epochs, visualization_loss_threshold,
                 gradient_clip_value, bptt_chunk_length=1000, print_param_on_training_end=False,
                 checkpoint_path=None):
        self._model = model
        self._optimizer = optimizer
        self._data_transform = data_transform
        self._objective = objective
        self._lr_sched = lr_sched
        
        self._train_loader = train_loader
        self._val_loaders = val_loaders
        if isinstance(self._val_loaders, list):
            self._val_loaders = {f'val{idx}': ldr for idx, ldr in enumerate(self._val_loaders)}

        self._gradient_clip_value = gradient_clip_value
        self._bptt_chunk_length = bptt_chunk_length
        assert gradient_clip_value is None or isinstance(gradient_clip_value, (int, float))

        self._checkpoint_path = checkpoint_path
        assert checkpoint_path is not None
        self._save_name = save_name
        self._device = device

        self._checkpoint_epochs = checkpoint_epochs
        self._print_interval = print_interval
        self._visualization_loss_threshold = visualization_loss_threshold
        self._visualization_epochs = visualization_epochs
        self._print_param_on_training_end = print_param_on_training_end
        
        # Initialize statistics variables @todo should we also add some KPI s.a. mIoU?
        self._stats = {}
        modes = ['train'] + list(self._val_loaders.keys())
        for mode in modes:
            for loss_key in objective.get_idfs():
                self._stats[f'{mode} {loss_key} loss'] = AverageMeter()

        self._epoch = 0

    def train(self, max_epochs):
        print(f"Training epochs {self._epoch + 1} to {max_epochs}. Moving model to {self._device}.")
        self._model.to(self._device)
        for epoch in range(self._epoch + 1, max_epochs + 1):
            self._epoch = epoch
            self._lr_sched.step(epoch)
            print(f"Starting epoch {epoch} with lr={self._lr_sched.get_lr()}")
            self._train_epoch()
            if self._epoch in self._checkpoint_epochs:
                print("Saving Checkpoint, current statistics are:")
                for key,val in self._stats.items():
                    strout = ["{:.3f}".format(elem) for elem in val.history]
                    print(key, strout, flush=True)
                self.save_checkpoint()
        print(f"Finished training!")
        if self._print_param_on_training_end:
            print("Final parameter statistics are:")
            for name, param in self._model.named_parameters():
                print_tensor_statistics(param, name)

    def _train_epoch(self):
        """Do one epoch of training and validation."""
        self._model.train(True)
        self._run_epoch(mode='train', data_loader=self._train_loader)

        self._model.train(False)
        with torch.no_grad():
            for loader_name, data_loader in self._val_loaders.items():
                self._run_epoch(mode=f'{loader_name}', data_loader=data_loader)

        # Update all stat values
        for stat_value in self._stats.values():
            if isinstance(stat_value, AverageMeter):
                stat_value.new_epoch()
                
    def _run_epoch(self, mode, data_loader):
        """ We expect to do Video Semantic Segmentation (VSS), Video 2D Object Detection (VOD),
            and Video Instance Segmentation (VIS).
        """
        for i, data in enumerate(data_loader):
            data = recursive_to(data, self._device)
            if hasattr(self._model, 'supervisor'):
                data = self._model.supervisor.augment_data(data, mode)

            assert set(data.keys()) <= {'images', 'ssannos', 'isannos', 'odannos', 'lbannos',
                                        'provides_ss', 'provides_is', 'provides_od', 'provides_lb',
                                        'identifier', 'active', 'video_id'}, data.keys()
            assert set(data.keys()) >= {'images', 'ssannos', 'isannos', 'odannos', 'lbannos',
                                        'provides_ss', 'provides_is', 'provides_od', 'provides_lb',
                                        'identifier', 'active'}, data.keys()
            
            self._optimizer.zero_grad()

            # For debugging purposes we feed in annotations
            visualize_this_iteration = (i == 0 and self._epoch in self._visualization_epochs)
            model_output, state = self._model(visualize=visualize_this_iteration, epoch=self._epoch, **data)

            if not (data['active'] == 1).any(): # No objects annotated in this batch, we skip it
                continue
            
            loss, partial_losses = self._objective(model_output, data, [mode, i])
            
            if mode == 'train':
                loss.backward()

               # ASSERTS
#                crash = False
#                for name, param in self._model.named_parameters():
#                    if param is None:
#                        print("parameter", name, "is None")
#                        continue
#                    if param.grad is None:
#                        continue
#                    if not torch.isfinite(param.grad).all():
#                        print(name, "gradient is nonfinite")
#                        crash = True

                self._optimizer.step()

                # ASSERTS
#                for name, param in self._model.named_parameters():
#                    if not torch.isfinite(param).all():
#                        print(name, param)
#                        crash = True
#                if crash:
#                    raise ValueError("Model parameter became infinite")

            partial_losses = {key: {'val': loss['val'].detach().to('cpu').item(), 'N': loss['N']}
                              for key, loss in partial_losses.items()}
            self.save_stats(partial_losses, model_output, data, mode)

            if visualize_this_iteration:
                self.visualize_batch(data, model_output, mode)

            del model_output, state

            if (i + 1) % self._print_interval == 0:
                loss_str = [(self._stats[f'{mode} {key} loss'].avg, key) for key in partial_losses.keys()]
                loss_str = [f"{val:.5f} ({name})" for val, name in loss_str]
                loss_str = "  ".join(loss_str)
                print(f"[{mode}: {self._epoch}, {i+1:4d}] Loss: {loss_str}")

        # end for
        loss_str = [(self._stats[f'{mode} {key} loss'].avg, key) for key in partial_losses.keys()]
        loss_str = [f"{val:.5f} ({name})" for val, name in loss_str]
        loss_str = "  ".join(loss_str)
        print(f"[{mode}: {self._epoch}] Loss: {loss_str}")
        return

    def save_checkpoint(self):
        """Saves a checkpoint of the network and other variables."""
        state = {
            'epoch': self._epoch,
            'net_type': type(self._model).__name__,
            'net': self._model.state_dict(),
            'optimizer' : self._optimizer.state_dict(),
            'stats' : self._stats,
            'device' : self._device,
        }
        file_path = '{}/{}_ep{:04d}.pth.tar'.format(self._checkpoint_path, self._save_name, self._epoch)
        torch.save(state, file_path)

    def load_checkpoint(self, checkpoint=None):
        """Loads a network checkpoint file.
        """
        if checkpoint is None: # Load most recent checkpoint
            checkpoint_list = sorted(glob.glob('{}/{}_ep*.pth.tar'.format(self._checkpoint_path, self._save_name)))
            if checkpoint_list:
                checkpoint_path = checkpoint_list[-1]
            else:
                print('No matching checkpoint file found')
                return
        elif isinstance(checkpoint, int): # Checkpoint is the epoch number
            
            checkpoint_path = '{}/{}_ep{:04d}.pth.tar'.format(self._checkpoint_path, self._save_name, checkpoint)
        elif isinstance(checkpoint, str): # checkpoint is the epoch file path
            checkpoint_path = os.path.expanduser(checkpoint)
        else:
            raise TypeError
        if not os.path.isfile(checkpoint_path):
            print(f"WARNING: Attempted to load checkpoint at epoch {checkpoint}, but it does not"
                  + " exist. Continuing without loading. If runfile is correctly set up, there will"
                  + " be an upcoming training stage that will begin from scratch.")
            return
        checkpoint_dict = torch.load(checkpoint_path)
        assert type(self._model).__name__ == checkpoint_dict['net_type'], 'Network is not of correct type'
        self._epoch = checkpoint_dict['epoch']
        self._model.load_state_dict(checkpoint_dict['net'])
        self._optimizer.load_state_dict(checkpoint_dict['optimizer'])
        self._stats = checkpoint_dict['stats']
        self._device = checkpoint_dict['device']
        self._lr_sched.step(self._epoch)
        print("Loaded: {}".format(checkpoint_path))

    def save_stats(self, partial_losses, model_output, data, mode):
        for name, loss in partial_losses.items():
            self._stats[f'{mode} {name} loss'].update(loss['val'], loss['N'])
        
    def visualize_batch(self, data, model_output, mode):
        images = data['images'].cpu().detach()
        B, L, _, H, W = images.size()
#        segannos = data['segannos'].cpu().detach()
        odannos = data['odannos'].cpu().detach() # (B,L,N,4)
#        lbannos = data['lbannos'].cpu().detach() # (B,N)
        isannos = data['isannos'].cpu().detach() # (B,L,H,W)
        active = (data['active'] == 1).cpu().detach() # (B,L,N)
        preds = recursive_detach(recursive_to(model_output['to_visualize'], 'cpu'))
        for b in range(B):
            for l in range(L):
                print(f"Anno for ({b},{l}) has {(active[b]>0).max(dim=0)[0].sum()} ODs > 0 (num objects)")
                odanno_fname = f"{self._save_name}_{mode}_b{b:03d}_l{l:03d}_anno.png"
                visible_anno_boxes = odannos[b,l][active[b,l]]
                boxlabels = active[b,l].nonzero().view(-1)
                debug_plot(image=images[b,l], seg=isannos[b,l], boxes=visible_anno_boxes,
                           boxlabels=boxlabels, fname=odanno_fname)
                for name, pred in preds.items():
                    if pred['boxlabels'] is None:
                        boxlabels = None
                    else:
                        boxlabels = pred['boxlabels'][b][l]
                    pred_fname = f"{self._save_name}_{mode}_b{b:03d}_l{l:03d}_{name}.png"
                    debug_plot(image=images[b,l], seg=pred['seg'][b,l], boxes=pred['boxes'][b][l],
                               boxlabels=boxlabels, fname=pred_fname)
        
class VOTVOSIdentityTransform(object):        
    def __call__(self, data):
        return data

