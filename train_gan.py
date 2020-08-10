import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
from random import shuffle
import matplotlib.pyplot as plt

from models import TransformerModel, MBGenerator, Discriminator
from reader import DataManager

from polyglot.mapping import Embedding, CaseExpander
from pathlib import Path

dm = DataManager()
train_d, test_d, dev_d = dm.get_batched_data(batch_sz=8)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# from polyglot.downloader import downloader
# downloader.download('embeddings2.en')

poly_path = '/polyglot_data/embeddings2/en/embeddings_pkl.tar.bz2'
embedding_path = str(Path.home()) + poly_path
polyglot_emb = Embedding.load(embedding_path)
polyglot_emb.apply_expansion(CaseExpander)
emsize = polyglot_emb.shape[1]
ntokens = len(dm.word2idx)  # the size of vocabulary

embeddings = torch.zeros(ntokens, emsize)
for w, i in dm.word2idx.items():
    if w in polyglot_emb:
        embeddings[i] = torch.tensor(polyglot_emb[w])
    else:
        embeddings[i] = torch.randn(emsize, dtype=torch.float)


def load_encoder_decoder():
    nhid = 200  # the dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 4  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 4  # the number of heads in the multiheadattention models
    dropout = 0.2  # the dropout value
    model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, embeddings, dropout).to(device)
    model.load_state_dict(torch.load("saved_models/20epochs_with_max_seq_len128.pt"))
    return model.to(device)


def random_words(count=3, batch_size=8):
    return torch.randint(1, ntokens, (count, batch_size), device=device)


encoder_decoder = load_encoder_decoder()
generator = MBGenerator(embeddings).to(device)
discriminator = Discriminator(64, batch_size=8).to(device)

# Initialize BCELoss function
criterion = nn.BCELoss()

# Used for outputting along training
fixed_noise = random_words()  # torch.randn(64, nz, 1, 1, device=device)
# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.
# Learning rate for optimizers
lr = 0.0002
# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))

# Training Loop

# Lists to keep track of progress
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
num_epochs = 5
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    # for i, data in enumerate(dataloader, 0):

    # model.train()  # Turn on the training mode
    total_loss = 0.
    start_time = time.time()
    batch_idxs = list(range(0, len(train_d)))
    shuffle(batch_idxs)
    for i, batch_idx in enumerate(batch_idxs):
        batch = train_d[batch_idx]

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        generator.zero_grad()
        # Format batch
        batch = batch.to(device)
        real_encoded = encoder_decoder.encode(batch)
        b_size = real_encoded.size(1)
        label = torch.full((b_size,), real_label, device=device)
        # Forward pass real batch through D
        output = discriminator(real_encoded).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = random_words()  # torch.randn(b_size, nz, 1, 1, device=device)
        # Generate fake image batch with G
        fake = generator(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = discriminator(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        generator.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = discriminator(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
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
                fake = generator(fixed_noise).detach().cpu()
            print(fake)

        iters += 1

plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses, label="G")
plt.plot(D_losses, label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()
