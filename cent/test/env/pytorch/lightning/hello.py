import os 
from torch import optim, nn, utils, Tensor 
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import lightning.pytorch as pl 


# define model
encoder = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))
decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))


class LitAutoEncoder(pl.LightningModule):
    def __init__(self, encoder, decoder, train_dataset):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.train_dataset = train_dataset
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop
        x, y = batch 
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)

        self.log("train_loss:", loss)
        return loss 
    
    def configure_optimizers(self): 
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer 

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                        num_workers=16,
                        persistent_workers=True,
                        batch_size=None,
                        pin_memory=True)

def train(dataset):
    autoencoder = LitAutoEncoder(encoder, decoder, dataset)
    # define dataset
    trainer = pl.Trainer(limit_train_batches=100, max_epochs=1)
    trainer.fit(model=autoencoder)

def use(dataset):
    checkpoint = "./lightning_logs/version_0/checkpoints/epoch=0-step=100.ckpt"
    autoencoder = LitAutoEncoder.load_from_checkpoint(checkpoint, 
        encoder=encoder, decoder=decoder, train_dataset=dataset)

    #embed 4 fake images
    fake_image_batch = Tensor(4, 28 * 28)
    embeddings = encoder(fake_image_batch)
    print("predictions (4 image embeddings):\n", embeddings, "\n")

