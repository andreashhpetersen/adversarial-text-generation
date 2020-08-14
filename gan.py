import copy
import json
import sys
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
from random import shuffle
import random
import matplotlib.pyplot as plt
from IPython.core.debugger import set_trace
import torch.nn.functional as F

from models import TransformerModel, SimpleGenerator, SimpleDiscriminator, EncoderDiscriminator, EncoderGenerator
from reader_sentences import DataManager

from polyglot.mapping import Embedding, CaseExpander
from pathlib import Path


def run(padding_eos, load=False):
    dm = DataManager(max_seq_len=15, normalize_to_max_seq_len_and_eos=True, eos=padding_eos)
    batch_size = 4
    train_d, test_d, dev_d = dm.get_batched_data(batch_sz=batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # from polyglot.downloader import downloader
    # downloader.download('embeddings2.en')

    poly_path = '/polyglot_data/embeddings2/en/embeddings_pkl.tar.bz2'
    embedding_path = str(Path.home()) + poly_path
    polyglot_emb = Embedding.load(embedding_path)
    polyglot_emb.apply_expansion(CaseExpander)
    emsize = polyglot_emb.shape[1]
    voc_sz = len(dm.word2idx)  # the size of vocabulary

    embeddings = torch.zeros(voc_sz, emsize)
    for w, i in dm.word2idx.items():
        if w in polyglot_emb:
            embeddings[i] = torch.tensor(polyglot_emb[w])
        else:
            embeddings[i] = torch.randn(emsize, dtype=torch.float)

    NHID = 200  # the dimension of the feedforward network model in nn.TransformerEncoder
    NLAYERS = 4  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    NHEAD = 4  # the number of heads in the multiheadattention models
    DROPOUT = 0.2  # the DROPOUT value

    suffix = ""
    if padding_eos:
        suffix = "-padding-eos"

    def load_source_model(device):
        model = TransformerModel(
            voc_sz, emsize, NHEAD, NHID, NLAYERS, embeddings, DROPOUT
        ).to(device)
        model.load_state_dict(
            torch.load(
                f"saved_models/transformer{suffix}.pt",
                map_location=device
            )
        )
        return model.to(device)

    def random_words(batch, repeat_words=True):
        rand = torch.randint(1, voc_sz, batch.shape, device=device)
        if repeat_words:
            for i in range(rand.shape[1]):
                for j in range(1, len(rand[:, i])):
                    repeat = random.randint(0, 1)
                    if repeat:
                        rand[j, i] = rand[j - 1, i]
        return rand

    def prepare_models(voc_sz, emsize, embeddings, device):
        source_model = load_source_model(device)
        generator = EncoderGenerator(
            voc_sz, emsize, NHEAD, NHID, NLAYERS, DROPOUT
        ).to(device)
        discriminator = EncoderDiscriminator(
            embeddings.size(1), dm.max_seq_len, NHEAD, NHID, NLAYERS, DROPOUT
        ).to(device)
        return source_model, generator, discriminator

    def ready_training(voc_sz, emsize, max_seq_len, embeddings, device, lr=0.0001, beta1=0.5):
        source_model, G, D = prepare_models(voc_sz, emsize, embeddings, device)
        noise = torch.randn((max_seq_len, 1, emsize)).to(device)
        optD = optim.Adam(D.parameters(), lr=lr, betas=(beta1, 0.999))
        optG = optim.Adam(G.parameters(), lr=lr, betas=(beta1, 0.999))
        return source_model, G, D, optD, optG, noise

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    # Initialize BCELoss function
    criterion = nn.BCELoss()

    source_model, G, D, optimizerD, optimizerG, fixed_noise = ready_training(
        voc_sz, emsize, dm.max_seq_len, embeddings, device
    )

    # Training Loop

    # Lists to keep track of progress
    G_losses = []
    D_losses = []
    iters = 0

    source_model.eval()
    source_model.requires_grad_(False)

    if not load:
        print("Starting Training Loop...")
        num_epochs = 10
        best_models_1 = (G, D)
        best_D_G_z1 = -math.inf
        best_models_2 = (G, D)
        best_D_G_z2 = -math.inf
        try:
            for epoch in range(num_epochs):
                G.train()
                D.train()
                total_loss = 0.
                start_time = time.time()
                batch_idxs = list(range(0, len(train_d)))
                shuffle(batch_idxs)
                for i, batch_idx in enumerate(batch_idxs):
                    batch = train_d[batch_idx].to(device)
                    batch_sz = batch.shape[1]

                    ############################
                    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                    ###########################
                    ## Train with all-real batch
                    D.zero_grad()
                    encoded_batch = source_model.encode(batch)
                    label = torch.full((batch_sz,), real_label, device=device)
                    output_real = D(encoded_batch)
                    errD_real = criterion(output_real, label)
                    errD_real.backward()

                    ## Train with all-fake batch from encoder
                    encoded_fakes = source_model.encode(random_words(batch))
                    label.fill_(fake_label)
                    output_fake = D(encoded_fakes)
                    errD_fake_encoded = criterion(output_fake, label)
                    errD_fake_encoded.backward()

                    D_x = torch.cat([output_real, output_fake]).mean().item()

                    ## Train with all-fake batch
                    label.fill_(fake_label)
                    noise = torch.randn((dm.max_seq_len, batch_sz, emsize), device=device)
                    fake = G(noise)
                    output = D(fake.detach()).view(-1)
                    errD_fake = criterion(output, label)
                    errD_fake.backward()
                    errD = errD_real + errD_fake

                    D_G_z1 = output.mean().item()

                    if (D_G_z1 > best_D_G_z1 and D_x > 0.35):
                        best_D_G_z1 = D_G_z1
                        best_models_1 = (copy.deepcopy(G), copy.deepcopy(D))

                    # Update D
                    optimizerD.step()

                    ############################
                    # (2) Update G network: maximize log(D(G(z)))
                    ###########################
                    G.zero_grad()

                    label.fill_(real_label)
                    output = D(fake).view(-1)
                    errG = criterion(output, label)
                    errG.backward()
                    D_G_z2 = output.mean().item()

                    if (D_G_z2 > best_D_G_z2 and D_x > 0.35):
                        best_D_G_z2 = D_G_z2
                        best_models_2 = (copy.deepcopy(G), copy.deepcopy(D))

                    # Update G
                    optimizerG.step()

                    # Output training stats
                    if i % 50 == 0:
                        print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                              % (epoch, num_epochs, i, len(batch_idxs),
                                 errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

                    # Save Losses for plotting later
                    G_losses.append(errG.item())
                    D_losses.append(errD.item())

                    # Check how the generator is doing by saving G's output on fixed_noise
                    if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(batch_idxs) - 1)):
                        with torch.no_grad():
                            G.eval()
                            fake = G(fixed_noise).detach()
                            decoded = source_model.decode(fake)
                            probs = F.softmax(decoded, dim=2).view(dm.max_seq_len, voc_sz)
                            print(dm.to_sentence(torch.argmax(probs, dim=1)))

                    iters += 1
        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')

        torch.save(G.state_dict(), f"saved_models/gan_generator{suffix}.pt")
        torch.save(D.state_dict(), f"saved_models/gan_discriminator{suffix}.pt")

        best_G1, best_D1 = best_models_1
        torch.save(best_G1.state_dict(), f"saved_models/best1_gan_generator{suffix}.pt")
        torch.save(best_D1.state_dict(), f"saved_models/best1_gan_discriminator{suffix}.pt")

        best_G2, best_D2 = best_models_2
        torch.save(best_G2.state_dict(), f"saved_models/best2_gan_generator{suffix}.pt")
        torch.save(best_D2.state_dict(), f"saved_models/best2_gan_discriminator{suffix}.pt")

        with open(f"saved_models/gan{suffix}.stats.json", "w") as file:
            file.write(json.dumps(
                {"G_losses": G_losses, "D_losses": D_losses, "best_D_G_z1": best_D_G_z1, "best_D_G_z2": best_D_G_z2}))

        plt.figure(figsize=(10, 5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(G_losses, label="G")
        plt.plot(D_losses, label="D")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
    else:
        G.load_state_dict(torch.load(f"saved_models/gan_generator{suffix}.pt"))
        D.load_state_dict(torch.load(f"saved_models/gan_discriminator{suffix}.pt"))

        best_G1, best_D1 = copy.deepcopy(G), copy.deepcopy(D)
        best_G1.load_state_dict(torch.load(f"saved_models/best1_gan_generator{suffix}.pt"))
        best_D1.load_state_dict(torch.load(f"saved_models/best1_gan_discriminator{suffix}.pt"))

        best_G2, best_D2 = copy.deepcopy(G), copy.deepcopy(D)
        best_G2.load_state_dict(torch.load(f"saved_models/best2_gan_generator{suffix}.pt"))
        best_D2.load_state_dict(torch.load(f"saved_models/best2_gan_discriminator{suffix}.pt"))

        def gen_noise():
            noise = torch.randn((dm.max_seq_len, 1, emsize), device=device)
            return noise

        def eval_gen(G, noise):
            with torch.no_grad():
                G.eval()
                fake = G(noise)
                decoded = source_model.decode(fake)
                probs = F.softmax(decoded, dim=2).view(dm.max_seq_len, voc_sz)
                print(dm.to_sentence(torch.argmax(probs, dim=1)))

        return G, D, best_G1, best_D1, best_G2, best_D2, gen_noise, eval_gen


if __name__ == '__main__':
    if sys.argv[1] == "True":
        G, D, best_G1, best_D1, best_G2, best_D2, gen_noise, eval_gen = run(False, load=True)

        noises = [gen_noise() for i in range(4)]

        for noise in noises:
            eval_gen(G, noise)
        print()
        for noise in noises:
            eval_gen(best_G1, noise)
        print()
        for noise in noises:
            eval_gen(best_G2, noise)
        print()

    elif sys.argv[1] == "False":
        run(False, load=False)
