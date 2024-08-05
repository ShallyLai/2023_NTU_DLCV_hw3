import torch
from torch import nn

class ViL(nn.Module):
    def __init__(self, encoder, decoder):
        super(ViL, self).__init__()
        
        self.encoder = encoder # from timm pre_trained weight 
        for param in self.encoder.parameters(): # freeze model
            param.requires_grad = False
            
        self.decoder = decoder # From TA

        # from huge or large to base
        self.linear_layer = nn.Linear(1024, 768) # huge: 1280, large: 1024

    def forward(self, images, captions):
        
        images_encoded = self.encoder.forward_features(images)  # need output tensor: 768
        images_encoded_linear = self.linear_layer(images_encoded) # (1, 257, 1280 or 1024) -> (1, 257, 769)
        decoder_outputs = self.decoder(captions, images_encoded_linear)

        return decoder_outputs
