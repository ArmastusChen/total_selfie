import torch

vae_path = 'pretrained_models/vae_20000.ckpt'
pretrained_model_path='pretrained_models/model.ckpt'
ckpt_file=torch.load(pretrained_model_path,map_location='cpu')
zero_data=torch.zeros(768, 1024 * 4)

# new_weight=torch.cat((ckpt_file['state_dict']['model.diffusion_model.input_blocks.0.0.weight'],zero_data),dim=1)


# for kk in ckpt_file['state_dict']:
    # if 'proj_out' in kk:
    #     print(kk)
        # ckpt_file['state_dict'][kk]=zero_data

vae_ckpt = torch.load(vae_path, map_location='cpu')


# replace ckpt_file with vae_ckpt
for kk in vae_ckpt['state_dict']:
        ckpt_file['state_dict'][kk]=vae_ckpt['state_dict'][kk]

ckpt_file['state_dict']['proj_out.weight']=zero_data
torch.save(ckpt_file,"pretrained_models/model_mlp_modified4_vae20000.ckpt")