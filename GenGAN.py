
import numpy as np
import cv2
import os
import pickle
import sys
import math

import matplotlib.pyplot as plt

from torchvision.io import read_image
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter


# sys.path.append('/kaggle/input/ativ-gan')
from VideoSkeleton import VideoSkeleton
from VideoReader import VideoReader
from Skeleton import Skeleton
from GenVanillaNN import * 

torch.set_default_dtype(torch.float32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class Discriminator(nn.Module):
    def __init__(self, ngpu=0):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu

        self.model = nn.Sequential(
            # Input: 3 x 64 x 64
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 2: 64 x 32 x 32
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 3: 128 x 16 x 16
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 4: 256 x 8 x 8
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Flatten: 512 x 4 x 4 -> 8192
            nn.Flatten(),
            
            # Fully connected layer to reduce to a single value
            nn.Linear(512 * 4 * 4, 1),
            nn.Sigmoid()  # Output probability between 0 and 1
        )


        
    def forward(self, input):
        return self.model(input)





class GenGAN():
    """ class that Generate a new image from videoSke from a new skeleton posture
       Fonc generator(Skeleton)->Image
    """
    def __init__(self, videoSke, loadFromFile=False):
        image_size = 64
        self.netG = GenNNSkeImToImage()
        self.netD = Discriminator()
        self.real_label = 1.
        self.fake_label = 0.
        src_transform = transforms.Compose([ SkeToImageTransform(image_size),
                                                 transforms.ToTensor(),
                                                 #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                 ])
        self.filename = 'data/Dance/DanceGenGAN.pth'
        tgt_transform = transforms.Compose(
                            [transforms.Resize((64, 64)),
                            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            transforms.CenterCrop(64),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ])
        self.dataset = VideoSkeletonDataset(videoSke, ske_reduced=True, target_transform=tgt_transform, source_transform=src_transform)
        self.dataloader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=1, shuffle=True)
        if loadFromFile and os.path.isfile(self.filename):
            print("GenGAN: Load=", self.filename, "   Current Working Directory=", os.getcwd())
            self.netG = torch.load(self.filename, map_location=torch.device('cpu'))


    def train(self, n_epochs=20):
    # Critère : Binary Cross-Entropy Loss
        criterion = nn.BCELoss()

        # Optimiseurs pour le discriminateur et le générateur
        optimizerD = optim.Adam(self.netD.parameters(), lr=0.0001, betas=(0.5, 0.999))
        optimizerG = optim.Adam(self.netG.parameters(), lr=0.0001, betas=(0.5, 0.999))

        print("Début de l'entrainement du GAN")
        num_minibatch = len(self.dataloader)

        for epoch in range(n_epochs):
            for i, (inputs, real_images) in enumerate(self.dataloader):
                batch_size = real_images.size(0)

                # Labels pour les vrais et faux exemples
                real_labels = torch.full((batch_size,), self.real_label, dtype=torch.float, device=real_images.device)
                fake_labels = torch.full((batch_size,), self.fake_label, dtype=torch.float, device=real_images.device)

                # ---- 1. Mise à jour de `netD` ----
                self.netD.zero_grad()

                real_images = real_images
                
                # Entrainer `netD` avec des vraies images
                outputs_real = self.netD(real_images).view(-1)
                loss_real = criterion(outputs_real, real_labels)
                loss_real.backward()

                # Entrainer `netD` avec des fausses images
                noise = inputs  # Les squelettes comme bruit
                fake_images = self.netG(noise)  # Générer des images à partir du générateur
                outputs_fake = self.netD(fake_images.detach()).view(-1)  # Detach pour ne pas calculer les gradients
                loss_fake = criterion(outputs_fake, fake_labels)
                loss_fake.backward()

                # Optimisation
                optimizerD.step()

                # ---- 2. Mise à jour de `netG` ----
                self.netG.zero_grad()

                # Essayer de tromper le discriminateur
                outputs_fake_for_G = self.netD(fake_images).view(-1)
                loss_G = criterion(outputs_fake_for_G, real_labels)
                loss_G.backward()

                # Optimisation
                optimizerG.step()

                # Affichage périodique des pertes
                if i % num_minibatch == num_minibatch - 1:
                    print(f'Epoch [{epoch+1}/{n_epochs}] | Batch [{i+1}/{len(self.dataloader)}] | '
                          f'Loss D: {loss_real.item() + loss_fake.item():.4f} | Loss G: {loss_G.item():.4f}')

        torch.save(self.netG, self.filename)
        print(f"Modèle GAN sauvegardé dans : {self.filename}")




    def generate(self, ske):
        """ Generate an image from a skeleton """
        # self.netG.eval()  # Mettre le générateur en mode évaluation

        # with torch.no_grad():  # Désactiver les gradients
        #     # Préparer le squelette
        #     ske_t = self.dataset.preprocessSkeleton(ske)
        #     ske_t_batch = ske_t.unsqueeze(0)  # Ajouter une dimension batch

        #     # Générer une image
        #     fake_image = self.netG(ske_t_batch)
        #     res = self.dataset.tensor2image(fake_image[0])  # Convertir en image
        #     return res
        ske_t = torch.from_numpy( ske.__array__(reduced=True).flatten() )
        ske_t = ske_t.to(torch.float32)
        ske_t = ske_t.reshape(1,Skeleton.reduced_dim,1,1) # ske.reshape(1,Skeleton.full_dim,1,1)
        normalized_output = self.netG(ske_t)
        res = self.dataset.tensor2image(normalized_output[0])
        return res




if __name__ == '__main__':
    force = False

    
    filename = "/kaggle/input/ativ-data/data/taichi1.mp4"
    print("GenGAN: Current Working Directory=", os.getcwd())
    print("GenGAN: Filename=", filename)

    targetVideoSke = VideoSkeleton(filename)

    # if False:
    if True:    # train or load
        # Train
        gen = GenGAN(targetVideoSke, False)
        gen.train(2000) #5) #200)
    else:
        gen = GenGAN(targetVideoSke, loadFromFile=True)    # load from file        




    for i in range(targetVideoSke.skeCount()):
        image = gen.generate(targetVideoSke.ske[i])
        #image = image*255
        nouvelle_taille = (256, 256) 
        image = cv2.resize(image, nouvelle_taille)
        cv2.imshow('Image', image)
        key = cv2.waitKey(-1)
        if key & 0xFF == ord('q'):
                    break
