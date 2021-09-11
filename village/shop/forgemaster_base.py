"""Main class, holding information about models and training/testing routines."""

import torch
import warnings
import time
import pickle

from ..utils import cw_loss, reverse_xent, reverse_xent_avg
from ..consts import NON_BLOCKING, BENCHMARK
torch.backends.cudnn.benchmark = BENCHMARK

class _Forgemaster():
    """Brew poison with given arguments.

    Base class.

    This class implements _forge(), which is the main loop for iterative poisoning.
    New iterative poisoning methods overwrite the _define_objective method.

    Noniterative poison methods overwrite the _forge() method itself.

    “Double, double toil and trouble;
    Fire burn, and cauldron bubble....

    Round about the cauldron go;
    In the poison'd entrails throw.”

    """

    def __init__(self, args, setup=dict(device=torch.device('cpu'), dtype=torch.float)):
        """Initialize a model with given specs..."""
        self.args, self.setup = args, setup
        self.retain = True if self.args.ensemble > 1 and self.args.local_rank is None else False
        self.stat_optimal_loss = None

    """ BREWING RECIPES """

    def forge(self, client, furnace):
        """Recipe interface."""
        if self.args.resume != '':
            resume_info = pickle.load( open( f'{self.args.resume}/info.pkl', 'rb'))
            global_poison_ids, idx = resume_info[0], resume_info[1] + 1
            if self.args.resume_idx is not None:
                idx = self.args.resume_idx
            #poison_ids, idx
            furnace.batched_construction_reset(global_poison_ids, idx)
        poison_delta = self._forge(client, furnace)

        return poison_delta

    def _forge(self, client, furnace):
        """Run generalized iterative routine."""
        print(f'Starting forgeing procedure ...')
        self._initialize_forge(client, furnace)
        poisons, scores = [], torch.ones(self.args.restarts) * 10_000

        for trial in range(self.args.restarts):
            poison_delta, target_losses = self._run_trial(client, furnace)
            scores[trial] = target_losses
            poisons.append(poison_delta.detach())
            if self.args.dryrun:
                break

        optimal_score = torch.argmin(scores)
        self.stat_optimal_loss = scores[optimal_score].item()
        print(f'Poisons with minimal target loss {self.stat_optimal_loss:6.4e} selected.')
        poison_delta = poisons[optimal_score]

        return poison_delta


    def _initialize_forge(self, client, furnace):
        """Implement common initialization operations for forgeing."""
        client.eval(dropout=True)
        # The PGD tau that will actually be used:
        # This is not super-relevant for the adam variants
        # but the PGD variants are especially sensitive
        # E.G: 92% for PGD with rule 1 and 20% for rule 2
        if self.args.attackoptim in ['PGD', 'GD']:
            # Rule 1
            #self.tau0 = self.args.eps / 255 / furnace.ds * self.args.tau * (self.args.pbatch / 512) / self.args.ensemble
            self.tau0 = self.args.eps / 255 / furnace.ds * self.args.tau
        elif self.args.attackoptim in ['momSGD', 'momPGD']:
            # Rule 1a
            self.tau0 = self.args.eps / 255 / furnace.ds * self.args.tau * (self.args.pbatch / 512) / self.args.ensemble
            self.tau0 = self.tau0.mean()
        else:
            # Rule 2
            self.tau0 = self.args.tau * (self.args.pbatch / 512) / self.args.ensemble



    def _run_trial(self, client, furnace):
        """Run a single trial."""
        poison_delta = furnace.initialize_poison()
        if self.args.full_data:
            dataloader = furnace.trainloader
        else:
            dataloader = furnace.poisonloader

        if self.args.attackoptim in ['Adam', 'signAdam', 'momSGD', 'momPGD']:
            # poison_delta.requires_grad_()
            if self.args.attackoptim in ['Adam', 'signAdam']:
                att_optimizer = torch.optim.Adam([poison_delta], lr=self.tau0, weight_decay=0)
            else:
                att_optimizer = torch.optim.SGD([poison_delta], lr=self.tau0, momentum=0.9, weight_decay=0)
            if self.args.scheduling:
                scheduler = torch.optim.lr_scheduler.MultiStepLR(att_optimizer, milestones=[self.args.attackiter // 2.667, self.args.attackiter // 1.6,
                                                                                            self.args.attackiter // 1.142], gamma=0.1)
            poison_delta.grad = torch.zeros_like(poison_delta)
            dm, ds = furnace.dm.to(device=torch.device('cpu')), furnace.ds.to(device=torch.device('cpu'))
            poison_bounds = torch.zeros_like(poison_delta)
        else:
            poison_bounds = None

        for step in range(self.args.attackiter):
            if step % 10 == 0:
                print(f'Step {step}')
            target_losses = 0
            poison_correct = 0
            for batch, example in enumerate(dataloader):
                if batch == 0:
                    start = time.time()
                elif batch % 100 == 0:
                    end = time.time()
                    avg = (end-start)/100
                    start = end
                    print(f'average time per epoch: {len(dataloader) * avg}')
                loss, prediction = self._batched_step(poison_delta, poison_bounds, example, client, furnace)
                target_losses += loss
                poison_correct += prediction

                if self.args.dryrun:
                    break

            # Note that these steps are handled batch-wise for PGD in _batched_step
            # For the momentum optimizers, we only accumulate gradients for all poisons
            # and then use optimizer.step() for the update. This is math. equivalent
            # and makes it easier to let pytorch track momentum.
            if self.args.attackoptim in ['Adam', 'signAdam', 'momSGD', 'momPGD']:
                if self.args.attackoptim in ['momPGD', 'signAdam']:
                    poison_delta.grad.sign_()
                att_optimizer.step()
                if self.args.scheduling:
                    scheduler.step()
                att_optimizer.zero_grad()
                with torch.no_grad():
                    # Projection Step
                    poison_delta.data = torch.max(torch.min(poison_delta, self.args.eps /
                                                            ds / 255), -self.args.eps / ds / 255)
                    poison_delta.data = torch.max(torch.min(poison_delta, (1 - dm) / ds -
                                                            poison_bounds), -dm / ds - poison_bounds)

            target_losses = target_losses / (batch + 1)
            poison_acc = poison_correct / len(dataloader.dataset)
            if step % (self.args.attackiter // 5) == 0 or step == (self.args.attackiter - 1):
                print(f'Iteration {step}: Target loss is {target_losses:2.4f}, '
                      f'Poison clean acc is {poison_acc * 100:2.2f}%')

            if self.args.step:
                if self.args.clean_grad:
                    client.step(furnace, None, self.targets, self.true_classes)
                else:
                    client.step(furnace, poison_delta, self.targets, self.true_classes)

            if self.args.dryrun:
                break

        return poison_delta, target_losses



    def _batched_step(self, poison_delta, poison_bounds, example, client, furnace):
        """Take a step toward minmizing the current target loss."""
        inputs, labels, ids = example
        inputs = inputs.to(**self.setup)
        labels = labels.to(dtype=torch.long, device=self.setup['device'], non_blocking=NON_BLOCKING)

        # Add adversarial pattern
        poison_slices, batch_positions = [], []
        for batch_id, image_id in enumerate(ids.tolist()):
            lookup = furnace.poison_lookup.get(image_id)
            if lookup is not None:
                poison_slices.append(lookup)
                batch_positions.append(batch_id)

        if len(batch_positions) > 0:
            delta_slice = poison_delta[poison_slices].detach().to(**self.setup)
            if self.args.clean_grad:
                delta_slice = torch.zeros_like(delta_slice)
            delta_slice.requires_grad_()
            poison_images = inputs[batch_positions]
            if self.args.recipe == 'poison-frogs':
                self.targets = inputs.clone().detach()
            inputs[batch_positions] += delta_slice

            # Perform differentiable data augmentation
            if self.args.paugment:
                inputs = furnace.augment(inputs, randgen=None)

            # Define the loss objective and compute gradients
            closure = self._define_objective(inputs, labels)
            loss, prediction = client.compute(closure)
            delta_slice = client.sync_gradients(delta_slice)

            if self.args.clean_grad:
                delta_slice.data = poison_delta[poison_slices].detach().to(**self.setup)

            # Update Step
            if self.args.attackoptim in ['PGD', 'GD']:
                delta_slice = self._pgd_step(delta_slice, poison_images, self.tau0, furnace.dm, furnace.ds)

                # Return slice to CPU:
                poison_delta[poison_slices] = delta_slice.detach().to(device=torch.device('cpu'))
            elif self.args.attackoptim in ['Adam', 'signAdam', 'momSGD', 'momPGD']:
                poison_delta.grad[poison_slices] = delta_slice.grad.detach().to(device=torch.device('cpu'))
                poison_bounds[poison_slices] = poison_images.detach().to(device=torch.device('cpu'))
            else:
                raise NotImplementedError('Unknown attack optimizer.')
        else:
            loss, prediction = torch.tensor(0), torch.tensor(0)

        return loss.item(), prediction.item()

    def _define_objective():
        """Implement the closure here."""
        def closure(model, criterion, *args):
            """This function will be evaluated on all GPUs."""  # noqa: D401
            raise NotImplementedError()
            return target_loss.item(), prediction.item()

    def _pgd_step(self, delta_slice, poison_imgs, tau, dm, ds):
        """PGD step."""
        with torch.no_grad():
            # Gradient Step
            if self.args.attackoptim == 'GD':
                delta_slice.data -= delta_slice.grad * tau
            else:
                delta_slice.data -= delta_slice.grad.sign() * tau

            # Projection Step
            delta_slice.data = torch.max(torch.min(delta_slice, self.args.eps /
                                                   ds / 255), -self.args.eps / ds / 255)
            delta_slice.data = torch.max(torch.min(delta_slice, (1 - dm) / ds -
                                                   poison_imgs), -dm / ds - poison_imgs)
        return delta_slice
