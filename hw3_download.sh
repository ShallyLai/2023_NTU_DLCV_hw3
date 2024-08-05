python3 -c "import clip; clip.load('ViT-B/32')"
python3 -c "import timm; timm.create_model('vit_large_patch14_clip_336.openai_ft_in12k_in1k', pretrained = True, num_classes = 0)"
gdown 10rZQ8Yr-3oRnV9y9Re6DowinK4ZtOcLn 
unzip ./p2_results.zip -d ./
rm p2_results.zip

