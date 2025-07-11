import random
from typing import TypedDict, List, Tuple, Callable, Self

import torch
from torch import nn
from torch.autograd import Variable
from torch.autograd import grad as torch_grad


from tqdm import tqdm

from python.gan.doko import DokoGenerator, DokoDiscriminator

# https://github.com/EmilienDupont/wgan-gp
class Trainer:
    G: DokoGenerator
    G_opt: torch.optim.Optimizer
    D: DokoDiscriminator
    D_opt: torch.optim.Optimizer

    batch_size: int

    losses: dict[str, List[float]]
    num_steps: int

    use_cuda: bool
    gp_weight: float
    critic_iterations: int
    print_every: int

    def __init__(self,
                 G: DokoGenerator,
                 D: DokoDiscriminator,
                 G_opt: torch.optim.Optimizer,
                 D_opt: torch.optim.Optimizer,

                 batch_size: int = 64,
                 gp_weight: float = 10,
                 critic_iterations: int = 5,
                 print_every: int = 50,
                 use_cuda: bool = False):
        self.G = G
        self.G_opt = G_opt
        self.D = D
        self.D_opt = D_opt
        self.batch_size = batch_size
        self.losses = {'G': [], 'D': [], 'GP': [], 'gradient_norm': []}
        self.num_steps = 0
        self.use_cuda = use_cuda
        self.gp_weight = gp_weight
        self.critic_iterations = critic_iterations
        self.print_every = print_every

        if self.use_cuda:
            self.G.cuda()
            self.D.cuda()

    def _gradient_penalty(self, states: torch.Tensor, real_data: torch.Tensor, generated_data: torch.Tensor):
        batch_size = real_data.size()[0]

        # Calculate interpolation
        alpha = torch.rand(batch_size, 1)
        alpha = alpha.expand_as(real_data)
        if self.use_cuda:
            alpha = alpha.cuda()
        interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data

        interpolated = Variable(interpolated, requires_grad=True)
        if self.use_cuda:
            interpolated = interpolated.cuda()

        # Calculate probability of interpolated examples
        prob_interpolated = self.D(states, interpolated)

        # Calculate gradients of probabilities with respect to examples
        grad_outputs = torch.ones(prob_interpolated.size())
        if self.use_cuda:
            grad_outputs = grad_outputs.cuda()
        gradients = torch_grad(outputs=prob_interpolated,
                               inputs=interpolated,
                               grad_outputs=grad_outputs,
                               create_graph=True, retain_graph=True)[0]

        gradients = gradients.view(batch_size, -1)
        self.losses['gradient_norm'].append(gradients.norm(2, dim=1).mean().data.item())

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return self.gp_weight * ((gradients_norm - 1) ** 2).mean()

    def _critic_train_iteration(
            self,
            # Paare aus echten States und Targets
            states: torch.Tensor,
            targets: torch.Tensor
    ):
        generated_data = self.sample_generator(
            states,
            states.size(0)
        )

        targets = Variable(targets)
        if self.use_cuda:
            targets = targets.cuda()

        d_real = self.D.forward(states, targets)
        d_generated = self.D(states, generated_data)

        gradient_penalty = self._gradient_penalty(states, targets, generated_data)
        self.losses['GP'].append(gradient_penalty.data.item())

        self.D_opt.zero_grad()
        d_loss = d_generated.mean() - d_real.mean() + gradient_penalty
        d_loss.backward()

        self.D_opt.step()

        self.losses['D'].append(d_loss.data.item())

    def _generator_train_iteration(self, states: torch.Tensor):
        """ """
        self.G_opt.zero_grad()

        # Get generated data
        batch_size = states.size()[0]
        generated_data = self.sample_generator(states, batch_size)

        # Calculate loss and optimize
        d_generated = self.D.forward(states, generated_data)
        g_loss = - d_generated.mean()
        g_loss.backward()
        self.G_opt.step()

        # Record loss
        self.losses['G'].append(g_loss.data.item())

    def _train_iteration(self, batched_data: List[Tuple[torch.Tensor, torch.Tensor]]):
        for i, batch in enumerate(batched_data):
            self.num_steps += 1

            (states, distributions) = batch

            self._critic_train_iteration(states, distributions)

            if self.num_steps % self.critic_iterations == 0:
                self._generator_train_iteration(states)

            if i % self.print_every == 0:
                print("Iteration {:<6} | D: {:<10.5f} | GP: {:<10.5f} | Gradient norm: {:<10.5f}".format(
                    i + 1, self.losses['D'][-1], self.losses['GP'][-1], self.losses['gradient_norm'][-1]
                ), end="")
                if self.num_steps > self.critic_iterations:
                    print(" | G: {:<10.5f}".format(self.losses['G'][-1]), end="")
                print()  # Zeilenumbruch

    def train(
            self,
            data_generator: Callable[[], List[Tuple[torch.Tensor, torch.Tensor]]],
            evaluator: Callable[[Callable[[Self], torch.Tensor], int], None],
            iterations: int
    ):
        for i in range(iterations):
            data = data_generator()

            self._train_iteration(
                data
            )

            self.G.eval()
            self.D.eval()
            evaluator(
                self,
                i
            )
            self.D.train()
            self.G.train()

            if i % 25 == 0:
                # save model
                torch.save(self.G.state_dict(), f"generator_256_6_{i}.pt")
                torch.save(self.D.state_dict(), f"discriminator_256_6_{i}.pt")

                print("Model saved")

    def sample_generator(self, states: torch.Tensor, num_samples: int):
        latent_samples = Variable(self.G.sample_latent(num_samples))

        if self.use_cuda:
            latent_samples = latent_samples.cuda()

        generated_data = self.G(states, latent_samples)

        return generated_data


