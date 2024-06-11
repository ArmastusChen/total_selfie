mkdir -p face_correction/ckpt
mv pretrained_weights/face_correction/00000196-checkpoint.pth.tar  face_correction/ckpt

mkdir -p selfie_conditioned_inpainting/ckpt
mv pretrained_weights/selfie_conditioned_inpainting/pbe_selfie2full  selfie_conditioned_inpainting/ckpt

mv pretrained_weights/seg/model_final.pth    seg

rm -rf pretrained_weights

