import torch
from PIL import Image
import os
import sys
import numpy as np
import json
import timm
from torchvision import transforms
from tqdm import tqdm
import torch.cuda.amp as amp

from tokenizer import BPETokenizer
from utils.p2_model import ViL
#from utils.decoder import Decoder, Config
#from utils.LoRA_decoder import Decoder, Config
from utils.adapter_decoder import Decoder, Config

image_path = sys.argv[1]  # "./hw3_data/p2_data/images/val" 
output_jason_path = sys.argv[2] # "./p2_pred_beam.json"
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
    max_length = 64

    encoder_model = timm.create_model("vit_large_patch14_clip_336.openai_ft_in12k_in1k", pretrained = True, num_classes = 0) 
    decoder_model = Decoder(Config(checkpoint = decoder_weight_path))
    
    # model specific transforms from timm
    data_config = timm.data.resolve_model_data_config(encoder_model)
    val_tfm = timm.data.create_transform(**data_config, is_training = False)
    
    model = ViL(encoder = encoder_model, decoder = decoder_model)

    pth_path = "./p2_results/p2_adapter/p2_7_adapter_1.pth"
    checkpoint = torch.load(pth_path, map_location = 'cpu')
    model.load_state_dict(checkpoint, strict = False)
    model.to(device)
    model.eval()

    tokenizer = BPETokenizer(encoder_file = "./encoder.json", vocab_file = "./vocab.bpe") 
    start_token = 50256 # "<|endoftext|>"
   
    image_names = [f for f in os.listdir(image_path) if f.endswith(".jpg")]

    avoid = [255, 3907, 8836, 16253, 18433, 20804, 22020, 25084, 27764, 29690, 29826, 34633, 36310, 39588, 40792, 41200, 48953, 49476]

    beam_width = 2
    voc_size = 2

    output_dict = {}
    for img in tqdm(image_names):
        image = Image.open(os.path.join(image_path, img)).convert("RGB")
        image = val_tfm(image) 
        image = image.unsqueeze(0)
        output = []
        caption = torch.zeros((1, max_length), dtype = torch.long) # set all zero
        caption[:, 0] = start_token 
        #print(caption)

        beam_captions = [caption.clone() for _ in range(beam_width)]
        beam_scores = [0.0] * beam_width
        flag = 0

        with amp.autocast():
            for j in range(max_length - 1):
                image = image.to(device)
                tmp_beam_captions = []
                tmp_beam_scores = []

                if j == 0:
                    caption = caption.to(device)
                    with torch.no_grad():
                        predictions = model(image, caption)
                        predictions = torch.nn.functional.log_softmax(predictions[:, j, :], dim=-1)
                    current_beam_scores, beam_indices = torch.topk(predictions, beam_width, dim=-1)
                    next_word_candidates = beam_indices

                    for beam_idx in range(beam_width):
                        current_caption = beam_captions[beam_idx].clone()
                        current_caption[:, j + 1] = next_word_candidates[0, beam_idx]

                        current_score = current_beam_scores[0, beam_idx].item()

                        tmp_beam_captions.append(current_caption)
                        tmp_beam_scores.append(current_score)
                    
                    beam_captions = tmp_beam_captions
                    beam_scores = tmp_beam_scores
                    #print(beam_captions)
                    #print(beam_scores)
                    continue

                for beam_idx in range(beam_width):
                    beam_captions[beam_idx] = beam_captions[beam_idx].to(device) 
                    #print(beam_captions[beam_idx])
                    with torch.no_grad():
                        predictions = model(image, beam_captions[beam_idx])
                        predictions = torch.nn.functional.log_softmax(predictions[:, j, :], dim=-1)
                    current_beam_scores, beam_indices = torch.topk(predictions, voc_size, dim=-1)
                    next_word_candidates = beam_indices[0]
                    #print(beam_indices)
                    #print(next_word_candidates)
                    #print(current_beam_scores)

                    if 50256 in next_word_candidates:
                        flag = 1
                        #print("END")
                        break

                    for candidate_idx in range(voc_size):
                        current_caption = beam_captions[beam_idx].clone()
                        current_caption[:, j + 1] = next_word_candidates[candidate_idx]

                        current_score = beam_scores[beam_idx] + current_beam_scores[0, candidate_idx].item()
                        #current_score = beam_scores[beam_idx] * current_beam_scores[0, candidate_idx].item()
                       
                        tmp_beam_captions.append(current_caption)
                        tmp_beam_scores.append(current_score) 

                if flag == 1:
                    flag = 0
                    break

                beam_captions = tmp_beam_captions
                beam_scores = tmp_beam_scores

                sorted_indices = np.argsort(beam_scores)[::-1] # 排序索引
                beam_captions = [beam_captions[i] for i in sorted_indices] # 照順序排
                beam_scores = [beam_scores[i] for i in sorted_indices]

                beam_captions = beam_captions[:beam_width]
                beam_scores = beam_scores[:beam_width]
                #print(beam_captions)
                #print(beam_scores)

        best_beam_idx = np.argmax(beam_scores)
        final_caption = beam_captions[best_beam_idx]

        output = final_caption.squeeze().cpu().numpy().tolist()
        output = [token for token in output if token != 0]
        output = output[1:]
 
        result = tokenizer.decode(np.array(output))
        result_list = result.capitalize()
        #print(result_list.capitalize())
        
        output_dict[img.split('.')[0]] = result_list.capitalize()
        
    with open(output_jason_path, 'w') as f:
        json.dump(output_dict, f, indent = 1)

if __name__ == "__main__":
    main()
    
