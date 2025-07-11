import numpy
import torch

from python.gan.Trainer import Trainer
from python.gan.doko.DokoDiscriminator import DokoDiscriminatorNetwork
from python.gan.doko.DokoGenerator import DokoGeneratorNetwork

from rs_doko_py_bridge import data_generator as doko_data_generator
from rs_doko_py_bridge import evaluate
from tqdm import tqdm

if __name__ == '__main__':
    latent_dim = 128
    hidden_dims = [2048, 2048, 2048, 2048]
    output_dim = 12
    num_epochs = 10_000_000
    batch_size = 64
    use_cuda = True

    noise_dim = 24

    device = torch.device("cuda" if use_cuda else "cpu")

    generator = DokoGeneratorNetwork(
        hidden_dims,
        24
    )
    discriminator = DokoDiscriminatorNetwork(
        hidden_dims
    )

    lr = 1e-4
    betas = (.9, .99)
    G_opt = torch.optim.Adam(generator.parameters(), lr=lr, betas=betas)
    D_opt = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=betas)

    trainer = Trainer(
        generator,
        discriminator,
        G_opt,
        D_opt,
        use_cuda=use_cuda,
        critic_iterations=40,
        gp_weight=5
    )

    def data_generator(batch_size: int):
        batched_training_data = []

        training_data = doko_data_generator(batch_size)

        training_data = [(torch.tensor(x).cuda(device), torch.tensor(y).cuda(device)) for x, y in training_data]


        return training_data

    def evaluation(trainer: Trainer, i: int):
        pass

    #
    # def data_generator(batch_size: int):
    #     training_data = []
    #     for i in tqdm(range(50000), desc="Generating training data"):
    #         # create some test data through real play
    #         sg = SimpleGame()
    #
    #         while not sg.is_finished():
    #             (obs, real) = sg.encode()
    #
    #             if use_cuda:
    #                 obs = obs.cuda()
    #                 real = real.cuda()
    #
    #             training_data.append((obs, real))
    #
    #             sg.play()
    #
    #     random.shuffle(training_data)
    #
    #     batched_training_data = []
    #     for i in tqdm(range(0, len(training_data), batch_size), desc="Batching training data"):
    #         obs_batch = torch.stack([obs for obs, _ in training_data[i:i + batch_size]])
    #         real_batch = torch.stack([real for _, real in training_data[i:i + batch_size]])
    #
    #         batched_training_data.append((obs_batch, real_batch))
    #
    #     return batched_training_data
    #
    # #https://www.reddit.com/r/learnmachinelearning/comments/17cma9u/some_things_i_learned_about_gan_training/?show=original
    #
    def evaluation(trainer: Trainer, i: int):
        def callback(
                batch_size: int,
                obs: numpy.array
        ):
            with torch.no_grad():
                obs = torch.tensor(obs).cuda()

                generated_data = (trainer
                                  .sample_generator(obs.unsqueeze(0).expand((batch_size, obs.size(0))), batch_size))
                return generated_data.cpu().numpy()

        evaluate(callback)

    def execute_discriminator(
            obs: numpy.array,
            real: numpy.array
    ):
        with torch.no_grad():
            obs = torch.tensor(obs).cuda()

            return trainer.D.forward(obs.unsqueeze(0), real.unsqueeze(0)).cpu().numpy()

    trainer.train(
        lambda: data_generator(batch_size),
        evaluation,
        num_epochs
    )

    pass