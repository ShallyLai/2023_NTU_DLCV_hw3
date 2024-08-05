import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
import os
import json
from torchvision import transforms

from tokenizer import BPETokenizer

# https://github.com/saahiluppal/catr/blob/master/datasets/coco.py
class CocoCaption(Dataset):
    def __init__(self, path, jason_path, limit, padding_length, tfm = None, train = True):
        super().__init__()
        self.path = path # image path
        self.transform = tfm
        self.train = train
        self.padding_length = padding_length 

        with open(jason_path, 'r') as file:
            ann = json.load(file)

        self.annot = [(self._process(val['image_id'], ann['images']), val['caption']) for val in ann['annotations']]
        if train == True:
            self.annot = self.annot
        else: # train == False
            self.annot = self.annot[: limit]
        
        # Tokenizer
        self.tokenizer = BPETokenizer(encoder_file = "./encoder.json", vocab_file = "./vocab.bpe") 
        #prompt = 'a kitchen with a sink and many cooking machines and a pot of food'
        #context = self.tokenizer.encode(prompt)
        #print(context) # 機器看的文字: [64, 9592, 351, 257, 14595, 290, 867, 10801, 8217, 290, 257, 1787, 286, 2057]
        #print(self.tokenizer.decode(context)) # 人類看得懂的文字: a kitchen with a sink and many cooking machines and a pot of food

    def _process(self, image_id, images):
        for image in images:
            if image['id'] == image_id:
                return image['file_name']
        raise AssertionError('No matching images!')

    def __len__(self):
        return len(self.annot)

    def __getitem__(self, index):
        file_name, caption = self.annot[index]
        image = Image.open(os.path.join(self.path, file_name)).convert("RGB")
        image = self.transform(image)
        
        # tokenizer from TA, start and end token: "<|endoftext|>" 50256
        caption_encoded = self.tokenizer.encode(caption) # list
        caption_encoded.insert(0, 50256)         
        caption_encoded.append(50256)
        #print(caption_encoded)

        label = self.tokenizer.encode(caption)
        label.append(50256)
        #print(label)
        
        # padding: add 50256, otherwise, kill 
        if len(caption_encoded) < self.padding_length:
            caption_encoded += [50256] * (self.padding_length - len(caption_encoded))
            label += [-100] * (self.padding_length - len(label))
        else:
            caption_encoded = caption_encoded[:self.padding_length]
            label = label[:self.padding_length]
        
        caption_ids = np.array(caption_encoded) # input_ids
        label_ids = np.array(label)
        #print(caption_ids)
        #print(label_ids)

        return image.squeeze(0), caption_ids, label_ids

def build_dataset(train, image_path, output_jason_path, tfm, padding_length):
    limit = -1

    if train == True:
        train_caption = CocoCaption(os.path.join(image_path, "train"), os.path.join(output_jason_path, "train.json"), 
                limit, padding_length, tfm = tfm, train = True)
        return train_caption

    else: # train == False
        val_caption = CocoCaption(os.path.join(image_path, "val"), os.path.join(output_jason_path, "val.json"), 
                limit, padding_length, tfm = tfm, train = False)
        return val_caption

