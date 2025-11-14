import os
import argparse
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.utils as vutils

# ------------------------
# Dataset
# ------------------------
class CaptchaDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.samples = []
        for fname in os.listdir(root):
            if fname.endswith('.png') and '-' in fname:
                label = fname.split('-')[0]
                self.samples.append((os.path.join(root, fname), label))

        if not self.samples:
            raise RuntimeError(f"No valid images found in {root}")

        self.chars = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.char_to_idx = {c: i+1 for i, c in enumerate(self.chars)}
        self.max_len = max(len(label) for _, label in self.samples)

        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((64, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def encode_label(self, label):
        vec = torch.zeros(self.max_len, dtype=torch.long)
        for i, c in enumerate(label):
            vec[i] = self.char_to_idx.get(c, 0)
        return vec

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("L")
        img = self.transform(img)
        label_tensor = self.encode_label(label)
        return img, label_tensor, label


# ------------------------
# Generator
# ------------------------
class Generator(nn.Module):
    def __init__(self, noise_dim, embed_dim):
        super().__init__()
        self.label_embed = nn.Embedding(63, embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(noise_dim + embed_dim, 256 * 8 * 16),
            nn.ReLU(True)
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        emb = self.label_embed(labels).mean(dim=1)
        x = torch.cat((noise, emb), dim=1)
        x = self.fc(x)
        x = x.view(x.size(0), 256, 8, 16)
        return self.deconv(x)


# ------------------------
# Discriminator
# ------------------------
class Discriminator(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.label_embed = nn.Embedding(63, embed_dim)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # Calculate feature map size: input (1, 64, 128)
        # After Conv 1 (stride 2): (64, 32, 64)
        # After Conv 2 (stride 2): (128, 16, 32)
        # After Conv 3 (stride 2): (256, 8, 16)
        # Total: 256 * 8 * 16 = 32768
        self.fc = nn.Linear(32768 + embed_dim, 1)

    def forward(self, img, labels):
        emb = self.label_embed(labels).mean(dim=1)
        feat = self.conv(img).view(img.size(0), -1)
        combined = torch.cat((feat, emb), dim=1)
        return self.fc(combined)


# ------------------------
# Checkpoint helpers
# ------------------------
def save_ckpt(epoch, G, D, optG, optD):
    torch.save({
        "epoch": epoch,
        "G": G.state_dict(),
        "D": D.state_dict(),
        "optG": optG.state_dict(),
        "optD": optD.state_dict()
    }, "captcha_gan_ckpt.pth")


def load_ckpt(device, G, D, optG, optD):
    ckpt = torch.load("captcha_gan_ckpt.pth", map_location=device)
    G.load_state_dict(ckpt["G"])
    D.load_state_dict(ckpt["D"])
    optG.load_state_dict(ckpt["optG"])
    optD.load_state_dict(ckpt["optD"])
    return ckpt["epoch"]


# ------------------------
# Training
# ------------------------
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = CaptchaDataset(args.data_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    noise_dim = 100
    embed_dim = 128
    G = Generator(noise_dim, embed_dim).to(device)
    D = Discriminator(embed_dim).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizerG = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizerD = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))

    os.makedirs("generated_gan", exist_ok=True)

    # resume if requested
    start_epoch = 0
    if args.resume:
        if os.path.exists("captcha_gan_ckpt.pth"):
            print("Resuming from checkpoint...")
            start_epoch = load_ckpt(device, G, D, optimizerG, optimizerD)
            print(f"Resumed at epoch {start_epoch}")
        else:
            print("No checkpoint found. Starting fresh.")

    # training loop
    for epoch in range(start_epoch, args.epochs):
        for imgs, label_tensors, raw_labels in dataloader:
            imgs = imgs.to(device)
            label_tensors = label_tensors.to(device)

            # --- Discriminator ---
            D.zero_grad()
            real_validity = D(imgs, label_tensors)
            real_loss = criterion(real_validity, torch.ones_like(real_validity))

            noise = torch.randn(imgs.size(0), noise_dim, device=device)
            fake_imgs = G(noise, label_tensors)
            fake_validity = D(fake_imgs.detach(), label_tensors)
            fake_loss = criterion(fake_validity, torch.zeros_like(fake_validity))

            D_loss = (real_loss + fake_loss) / 2
            D_loss.backward()
            optimizerD.step()

            # --- Generator ---
            G.zero_grad()
            gen_validity = D(fake_imgs, label_tensors)
            G_loss = criterion(gen_validity, torch.ones_like(gen_validity))
            G_loss.backward()
            optimizerG.step()

        print(f"[Epoch {epoch+1}/{args.epochs}] D_loss={D_loss.item():.4f}, G_loss={G_loss.item():.4f}")

        # save a sample
        sample_text = "123456"
        sample_label = torch.zeros(dataset.max_len, dtype=torch.long)
        for i, c in enumerate(sample_text):
            if i >= dataset.max_len:
                break
            sample_label[i] = dataset.char_to_idx.get(c, 0)
        sample_label = sample_label.unsqueeze(0).to(device)

        sample_noise = torch.randn(1, noise_dim, device=device)
        with torch.no_grad():
            img = G(sample_noise, sample_label)

        vutils.save_image(img, f"generated_gan/sample_{epoch+1:03d}.png", normalize=True)

        # save checkpoint
        save_ckpt(epoch + 1, G, D, optimizerG, optimizerD)

    torch.save(G.state_dict(), "captcha_gan_G.pt")
    torch.save(D.state_dict(), "captcha_gan_D.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    train(args)
