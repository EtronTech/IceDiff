# Swin U-Net with TSAM
from .Modules.swin_transformer_v2_1 import SwinTransformerV2_Encoder as Encoder
from .Modules.swin_transformer_v2_1 import SwinTransformerV2_Decoder as Decoder


import torch
import torch.nn as nn


class SwinV2_Unet(nn.Module):
    def __init__(self,config,
                 ckpts = None):
        super().__init__()       
        config = config.PRED_FOR_SR
        self.encoder = Encoder(img_size=config.DATA.IMG_SIZE,
                             patch_size=config.MODEL.SWINV2.PATCH_SIZE,
                             in_chans=config.MODEL.SWINV2.IN_CHANS,
                             embed_dim=config.MODEL.SWINV2.EMBED_DIM,
                             depths=config.MODEL.SWINV2.DEPTHS,
                             num_heads=config.MODEL.SWINV2.NUM_HEADS,
                             window_size=config.MODEL.SWINV2.WINDOW_SIZE,
                             mlp_ratio=config.MODEL.SWINV2.MLP_RATIO,
                             qkv_bias=config.MODEL.SWINV2.QKV_BIAS,
                             drop_rate=config.MODEL.DROP_RATE,
                             drop_path_rate=config.MODEL.DROP_PATH_RATE,
                             ape=config.MODEL.SWINV2.APE,
                             patch_norm=config.MODEL.SWINV2.PATCH_NORM,
                             use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                             pretrained_window_sizes=config.MODEL.SWINV2.PRETRAINED_WINDOW_SIZES)

        self.decoder = Decoder(img_size=config.DATA.IMG_SIZE,
                             patch_size=config.MODEL.SWINV2.PATCH_SIZE,
                             in_chans=config.MODEL.SWINV2.IN_CHANS,
                             out_chans=config.MODEL.SWINV2.OUT_CHANS,
                             embed_dim=config.MODEL.SWINV2.EMBED_DIM,
                             depths=config.MODEL.SWINV2.DEPTHS,
                             num_heads=config.MODEL.SWINV2.NUM_HEADS,
                             window_size=config.MODEL.SWINV2.WINDOW_SIZE,
                             mlp_ratio=config.MODEL.SWINV2.MLP_RATIO,
                             qkv_bias=config.MODEL.SWINV2.QKV_BIAS,
                             drop_rate=config.MODEL.DROP_RATE,
                             drop_path_rate=config.MODEL.DROP_PATH_RATE,
                             ape=config.MODEL.SWINV2.APE,
                             patch_norm=config.MODEL.SWINV2.PATCH_NORM,
                             use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                             pretrained_window_sizes=config.MODEL.SWINV2.PRETRAINED_WINDOW_SIZES)

        if ckpts is None:
            pass
        else:
            # Load Pre-trained Independently Ckpt
            try:
                self.encoder.load_state_dict(torch.load(ckpts[0]))
                self.decoder.load_state_dict(torch.load(ckpts[1]))
            except:
                raise ValueError("Invalid Parameter Checkpoints")
            else:
                pass
    
    def forward(self,X):
        feats, skip_connections = self.encoder(X)
        forecast_leads = self.decoder(feats,skip_connections)
        return forecast_leads

    def count_parameters(self):
        '''
        Count Model Size
        '''
        encoder_params = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        decoder_params = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        ae_params = encoder_params + decoder_params

        # Return in Millions
        return encoder_params/1000000, decoder_params/1000000, ae_params/1000000
       

    def save_parameters(self,config,epoch):
        '''
        Save Encoder and Decoder Parameters Independently
        '''
        # torch.save(self.encoder,config.TRAIN.SAVE_CKPT_DIR + '/encoder_' + str(epoch) + '.pth')
        torch.save(self.encoder.state_dict(),config.TRAIN.SAVE_CKPT_DIR + '/unet_encoder_' + str(epoch) + '.pth')
        torch.save(self.decoder.state_dict(),config.TRAIN.SAVE_CKPT_DIR + '/unet_decoder_' + str(epoch) + '.pth')
    
    def load_state_dicts(self,ae_state_dicts):
        '''
        load parameters from trained checkpoints
        '''
        self.encoder.load_state_dict(ae_state_dicts['encoder'])
        self.decoder.load_state_dict(ae_state_dicts['decoder'])
    
    def get_state_dicts(self):
        ae_state_dict = {}
        ae_state_dict['encoder'] = self.encoder.state_dict()
        ae_state_dict['decoder'] = self.decoder.state_dict()
        return ae_state_dict

    @property
    def get_encoder(self):
        return self.encoder
    
    @property
    def get_decoder(self):
        return self.decoder
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"cpb_mlp", "logit_scale", 'relative_position_bias_table'}