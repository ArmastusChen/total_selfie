name=demo
gpu=0
# settings
output_dir='outputs'
mask_type=rectangle
mask_expand_ratio=1.1
guidance_scale=5.0
start_ind=0
end_ind=20

PBE_MODEL_ID='./selfie_conditioned_inpainting/ckpt/pbe_selfie2full'


: '
Sec 3.2: Per-Capture Preprocessing and Fine-Tuning
'
# STEP 1: Data preprocessing, it resizes image to 512x512, undistorts the face, crop the shoes region, infer the mask from the target pose image. 
# The outputs are in data/$name/processed
CUDA_VISIBLE_DEVICES=$gpu python inference/prepare_image.py   --gpu $gpu --name $name 

# STEP 2: Direct inference from trained inpainting model, without any pose condition and fine-tuning. This is to produce images as reference to place the selfie images for fine-tuning.
# The outputs are in $output_dir/inference/results/no_cond
CUDA_VISIBLE_DEVICES=$gpu python inference/inference.py  --output_dir $output_dir  --model_id $PBE_MODEL_ID  --name $name   --mask_type $mask_type   --mask_expand_ratio $mask_expand_ratio   --guidance_scale $guidance_scale   --start_ind $start_ind --end_ind $end_ind 

# STEP 3: Place the selfie images on the reference images yo generate the augmented images for fine-tuning. The placement stratetgy is based on the generated unconditional full body images.
# The outputs are in $output_dir/fine_tune_pbe_data
CUDA_VISIBLE_DEVICES=$gpu python fine_tuning/generate_augmentation_pbe.py  --model_id $PBE_MODEL_ID   --output_dir $output_dir  --name $name   --mask_type $mask_type   --mask_expand_ratio $mask_expand_ratio   --guidance_scale $guidance_scale   



# STEP 4: Fine tune the selfie to full body inpainting model with the augmented images.
# The outputs are in $output_dir/fine_tune_pbe_ckpt
CUDA_VISIBLE_DEVICES=$gpu python fine_tuning/fine_tune_pbe.py    --use_mask_loss  --model_id $PBE_MODEL_ID    --output_dir $output_dir   --name $name   --mask_type $mask_type   --mask_expand_ratio $mask_expand_ratio   --guidance_scale $guidance_scale    --max_train_steps 800

# settings
model_id="./$output_dir/fine_tune_pbe_ckpt/$name/pbe_selfie2full/$mask_type/$mask_expand_ratio/$guidance_scale/checkpoint-800"
guidance_scale=3.0
dilate_size=21
seed=101
mode=canny
edit_epoch=300

# STEP 5: Inference using the fine-tuned model.
# The outputs are in $output_dir/inference/results/canny
CUDA_VISIBLE_DEVICES=$gpu python inference/inference.py   --seed $seed  --blending_step 100  --output_dir $output_dir   --dilate_size $dilate_size  --mode $mode  --model_id $model_id --name $name   --mask_type $mask_type   --mask_expand_ratio $mask_expand_ratio   --guidance_scale $guidance_scale 


: '
Appearance Refinement for face and shoes
'

# STEP 6: Generate the augmented images for dreambooth.
# The outputs are in $output_dir/fine_tune_db_data
CUDA_VISIBLE_DEVICES=$gpu python fine_tuning/generate_augmentation_db.py --output_dir $output_dir  --name $name   --mask_type $mask_type   --mask_expand_ratio $mask_expand_ratio   --guidance_scale $guidance_scale 

# STEP 7: Fine tune a dreambooth model with the augmented images.
# The outputs are in $output_dir/fine_tune_db_ckpt
CUDA_VISIBLE_DEVICES=$gpu python fine_tuning/fine_tune_db.py  --output_dir $output_dir   --name $name   --mask_type $mask_type   --mask_expand_ratio $mask_expand_ratio   --guidance_scale $guidance_scale 

# STEP 8: Appearance refinement for face and shoes using dreambooth model.
# The outputs are in $output_dir/results_final.  If you want to refine the hands, you could use Stable Diffusion Webui.
CUDA_VISIBLE_DEVICES=$gpu python inference/inpaint.py   --strengths 0.4,0.1   --seeds  10,10 --output_dir $output_dir  --dilate_size $dilate_size  --mode $mode   --model_id $model_id    --name $name   --mask_type $mask_type   --mask_expand_ratio $mask_expand_ratio   --guidance_scale $guidance_scale   --edit_epoch $edit_epoch --selected_ind $seed


