src_path=models/selfie/2024-06-13T11-11-43_selfie_v2/checkpoints/epoch=000009.ckpt
dump_path=ckpt/pbe_selfie2full

python scripts/convert_original_stable_diffusion_to_diffusers.py --checkpoint_path $src_path  --dump_path $dump_path --num_in_channels 9 --pipeline_type PaintByExample