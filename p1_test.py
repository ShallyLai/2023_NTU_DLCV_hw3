import torch
import clip
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import os
import sys

myseed = 53  # set a random seed for reproducibility
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(myseed)
    torch.cuda.manual_seed_all(myseed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

image_path = sys.argv[1] # "./hw3_data/p1_data/val" 
id2label_path = sys.argv[2] # "./hw3_data/p1_data/id2label.json"
output_path = sys.argv[3] # "./p1_predict.csv"

def main():
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, preprocess = clip.load("ViT-B/32", device = device)

    image_names = [f for f in os.listdir(image_path) if f.endswith(".png")]

    with open(id2label_path, 'r') as file:
        id2label = json.load(file)
    labels = [v for k, v in id2label.items()]

    indices = []
    for image in tqdm(image_names):
        image = preprocess(Image.open(os.path.join(image_path, image))).unsqueeze(0).to(device)
        text = torch.cat([clip.tokenize(f"a photo of a {c}") for c in labels]).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)

            image_features /= image_features.norm(dim = -1, keepdim = True)
            text_features /= text_features.norm(dim = -1, keepdim = True)
        
        similarity = (100.0 * image_features @ text_features.T).softmax(dim = -1)
        value, index = similarity[0].topk(1) # topk(n): Pick the top n most similar labels for the image
        indices.append(index)

    predict = []
    for index in indices:
        predict.append(index.item())

    df = pd.DataFrame({'filename': image_names, 'label': predict})
    df.to_csv(output_path, index = False)

if __name__ == "__main__":
    main()

