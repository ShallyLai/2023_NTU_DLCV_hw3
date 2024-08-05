import torch
from PIL import Image
import os
import sys
import numpy as np
import json
import timm
from torchvision import transforms
from tqdm import tqdm

from tokenizer import BPETokenizer
from utils.p2_model import ViL
#from utils.decoder import Decoder, Config
#from utils.LoRA_decoder import Decoder, Config
from utils.adapter_decoder import Decoder, Config

image_path = sys.argv[1] # "./hw3_data/p2_data/images/val"
output_jason_path = sys.argv[2] # "./p2_pred.json"
decoder_weight_path = sys.argv[3] # "./hw3_data/p2_data/decoder_model.bin"

myseed = 53  # set a random seed for reproducibility
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(myseed)
    torch.cuda.manual_seed_all(myseed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def main():
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_length = 96

    encoder_model = timm.create_model("vit_large_patch14_clip_336.openai_ft_in12k_in1k", pretrained = True, num_classes = 0)
    decoder_model = Decoder(Config(checkpoint = decoder_weight_path))
    
    # model specific transforms from timm
    data_config = timm.data.resolve_model_data_config(encoder_model)
    val_tfm = timm.data.create_transform(**data_config, is_training = False)
    
    model = ViL(encoder = encoder_model, decoder = decoder_model)

    pth_path = "./p2_results/p2_adapter/p2_7_adapter_1.pth"
    checkpoint = torch.load(pth_path, map_location = 'cpu')
    print(sum([p.numel() for n, p in checkpoint.items()]))
    model.load_state_dict(checkpoint, strict = False)
    model.to(device)
    model.eval()

    tokenizer = BPETokenizer(encoder_file = "./encoder.json", vocab_file = "./vocab.bpe") 
    start_token = 50256 # "<|endoftext|>"
   
    image_names = [f for f in os.listdir(image_path) if f.endswith(".jpg")]

    avoid = [255, 3907, 8836, 16253, 18433, 20804, 22020, 25084, 27764, 29690, 29826, 34633, 36310, 39588, 40792, 41200, 48953, 49476]

    output_dict = {}
    for img in tqdm(image_names):
        image = Image.open(os.path.join(image_path, img)).convert("RGB")
        image = val_tfm(image) 
        image = image.unsqueeze(0)
        output = []
        caption = torch.zeros((1, max_length), dtype = torch.long) # set all zero
        caption[:, 0] = start_token 
        #print(caption)

        for j in range(max_length - 1):
            image = image.to(device)
            caption = caption.to(device)
            
            with torch.no_grad():
                predictions = model(image, caption)
            predictions = predictions[:, j, :]
            predicted_id = torch.argmax(predictions, axis = -1)

            # drop padding
            if predicted_id[0].item() == 50256:
                break
            if j == 71:
                break

            if predicted_id[0].item() in avoid:
                _, indices = torch.topk(predictions, 2, dim=-1)
                predicted_id = indices[:, :, 1]

            caption[:, j+1] = predicted_id[0]
            output.append(predicted_id[0].item())
        #print(output)

        result = tokenizer.decode(np.array(output))
        result_list = result.capitalize()
        #print(result_list.capitalize())
        
        output_dict[img.split('.')[0]] = result_list.capitalize()
        
    with open(output_jason_path, 'w') as f:
        json.dump(output_dict, f, indent = 1)

if __name__ == "__main__":
    main()
    
