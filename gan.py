import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
from random import shuffle
import matplotlib.pyplot as plt
from IPython.core.debugger import set_trace
import torch.nn.functional as F

from models import TransformerModel, SimpleGenerator, SimpleDiscriminator, EncoderDiscriminator, EncoderGenerator
from reader import DataManager

from polyglot.mapping import Embedding, CaseExpander
from pathlib import Path

dm = DataManager()
batch_size = 8
train_d, test_d, dev_d = dm.get_batched_data(batch_sz=batch_size)
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


nhid = 200  # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 4  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 4  # the number of heads in the multiheadattention models
dropout = 0.2  # the dropout value

def load_source_model():
    model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, embeddings, dropout).to(device)
    model.load_state_dict(torch.load("saved_models/20epochs_with_max_seq_len128.pt"))
    return model.to(device)


# def random_words(count=3):
#     return torch.randint(1, ntokens, (count, batch_size), device=device)

def random_words(batch):
    rand = torch.randint(1, ntokens, batch.size(), device=device)
    for i in range(batch.size(0)):
        for j in range(batch.size(1)):
            if batch[i][j] == 0:
                rand[i][j] = 0
    return rand

source_model = load_source_model()
generator = EncoderGenerator(ntokens, emsize, nhead, nhid, nlayers, embeddings, dropout).to(device)
# generator = SimpleGenerator(embeddings).to(device)
discriminator = EncoderDiscriminator(embeddings.size(1), dm.max_seq_len, nhead, nhid, nlayers, dropout).to(device)
# discriminator = SimpleDiscriminator(embeddings.size(1), dm.max_seq_len).to(device)

# Initialize BCELoss function
criterion = nn.BCELoss()

# Used for outputting along training
fixed_noise = torch.randint(1, ntokens, (dm.max_seq_len, 1)).to(device) # torch.randn(64, nz, 1, 1, device=device)
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
num_epochs = 20
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
        batch = train_d[batch_idx].to(device)
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        generator.zero_grad()
        # Format batch
        encoded_batch = source_model.encode(batch)
        batch_size = encoded_batch.size(1)
        # a list of 1's matching the length of the batch size, e.g. we expect each sentence in batch to be real
        label = torch.full((batch_size,), real_label, device=device)
        # Forward pass real batch through D
        output = discriminator(encoded_batch)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = random_words(batch)  # torch.randn(b_size, nz, 1, 1, device=device)
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
                fake = generator(fixed_noise).detach()
                decoded = source_model.decode(fake)
                probs = F.softmax(decoded, dim=2).view(dm.max_seq_len, ntokens)
                print(dm.to_sentence(torch.argmax(probs, dim=1)))

        iters += 1

plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses, label="G")
plt.plot(D_losses, label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()
