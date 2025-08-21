from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import torch
import torch.nn as nn
from Gan_def import generator, discriminator, device, latent_dim
import torchvision

# حذف این خط: latent_dim = 100

class BoreholeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        # فقط فایل‌های با پسوند .png را جمع‌آوری می‌کنیم
        self.img_paths = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir) 
                         if fname.lower().endswith('.png')]
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

# تبدیلات داده برای افزایش تنوع
transform = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.RandomHorizontalFlip(p=0.3),  # وارونه کردن افقی
    transforms.ColorJitter(brightness=0.1, contrast=0.1),  # تغییر روشنایی و کنتراست
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

dataset = BoreholeDataset("synthetic_borehole_dataset", transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Loss & Optimizers
adversarial_loss = nn.BCELoss()
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002)

# Training loop
epochs = 100
for epoch in range(epochs):
    for imgs in dataloader:
        real_imgs = imgs.to(device)
        batch_size = real_imgs.size(0)

        # Labels
        valid = torch.ones((batch_size, 1), device=device)
        fake = torch.zeros((batch_size, 1), device=device)

        # Train Generator
        optimizer_G.zero_grad()
        z = torch.randn(batch_size, latent_dim, device=device)
        gen_imgs = generator(z)
        validity = discriminator(gen_imgs)
        g_loss = adversarial_loss(validity, valid)
        g_loss.backward()
        optimizer_G.step()

        # Train Discriminator
        optimizer_D.zero_grad()
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

    print(f"[Epoch {epoch}/{epochs}] D_loss: {d_loss.item():.4f} | G_loss: {g_loss.item():.4f}")
    
    if epoch % 10 == 0:
        os.makedirs("generated", exist_ok=True)
        torchvision.utils.save_image(gen_imgs[:8], f"generated/epoch_{epoch}.png", normalize=True)
