import torch
import clip
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import os
from torch.utils.data import DataLoader
import torch.nn as nn
import math
import sys
import timm
from torch.cuda.amp import autocast as autocast

from utils.adapter_decoder import Decoder, Config
from tokenizer import BPETokenizer
from p2_dataset import build_dataset
from utils.p2_model import ViL

image_path = "./hw3_data/p2_data/images" # train and val
output_jason_path = "./hw3_data/p2_data" # train.json and val.json
decoder_weight_path = "./hw3_data/p2_data/decoder_model.bin"

myseed = 53  # set a random seed for reproducibility
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(myseed)
    torch.cuda.manual_seed_all(myseed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def main():
   
    epochs = 10

    batch_size = 32

    padding_length = 96
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    encoder_model = timm.create_model("vit_large_patch14_clip_336.openai_ft_in12k_in1k", pretrained = True, num_classes = 0)
    decoder_model = Decoder(Config(checkpoint = decoder_weight_path)) # decoder from TA
    for param in decoder_model.parameters(): # freeze decoder
        param.requires_grad = False
    
    # model specific transforms from timm
    data_config = timm.data.resolve_model_data_config(encoder_model)
    train_tfm = timm.data.create_transform(**data_config, is_training = True)
    val_tfm = timm.data.create_transform(**data_config, is_training = False)
    
    # Visual language model
    model = ViL(encoder = encoder_model, decoder = decoder_model)
    for block in model.decoder.transformer.h:
        for param in block.multihead_attention.parameters():
            param.requires_grad = True
        if hasattr(block, 'adapter'):
            for param in block.adapter.parameters():
                param.requires_grad = True
    trainable_weights = [name for name, param in model.named_parameters() if param.requires_grad == True]
    print("Total params:", sum(p.numel() for p in model.parameters() if p.requires_grad == True), "\n")
    model.to(device)
    #print(model.decoder)

    train_set = build_dataset(True, image_path, output_jason_path, train_tfm, padding_length)
    val_set = build_dataset(False, image_path, output_jason_path, val_tfm, padding_length)

    sampler_train = torch.utils.data.RandomSampler(train_set)
    sampler_val = torch.utils.data.SequentialSampler(val_set)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, batch_size, drop_last = True)

    train_loader = DataLoader(train_set, batch_sampler = batch_sampler_train, num_workers = 4)
    val_loader = DataLoader(val_set, batch_size, sampler = sampler_val, drop_last = False, num_workers = 4)

    tokenizer = BPETokenizer(encoder_file = "./encoder.json", vocab_file = "./vocab.bpe") 
    
    optimizer = torch.optim.AdamW(model.parameters(), lr = 0.0003, weight_decay = 5e-4)
    criterion = nn.CrossEntropyLoss()
    #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 20)
    scaler = torch.cuda.amp.GradScaler()
    
    clip_max_norm = 0.1
    for epoch in range(epochs):
        print(f"Start epoch {epoch + 1}")
        model.train()
        criterion.train()

        epoch_loss = 0.0
        total = len(train_loader)

        with tqdm(total = total) as pbar:
            for images, caps, labels in train_loader:
                images = images.to(device)
                caps = caps.to(device)
                labels = labels.to(device)
                
                with autocast():
                    outputs = model(images, caps[:, :-1])
                    loss = criterion(outputs.permute(0, 2, 1), labels[:, :-1])
                loss_value = loss.item()
                epoch_loss += loss_value

                if not math.isfinite(loss_value):
                    print(f'Loss is {loss_value}, stopping training')
                    sys.exit(1) 

                optimizer.zero_grad()
                #loss.backward()
                if clip_max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
                #optimizer.step()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                pbar.update(1)
        epoch_loss = epoch_loss / total
        #lr_scheduler.step()
        print(f"epoch: {epoch + 1} | Training Loss: {epoch_loss}\n")

        #torch.save(model.state_dict(), f"./p2_results/p2_adapter_1/p2_{epoch + 1}_checkpoint_adapter_1.pth")
        save_weights = {k: v for k, v in model.state_dict().items() if k in trainable_weights}
        torch.save(save_weights, f"./p2_results/p2_adapter_1/p2_{epoch + 1}_adapter_1.pth")
        
if __name__ == "__main__":
    main()

