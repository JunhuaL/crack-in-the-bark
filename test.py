from datasets import ImgDirDataset, LatentDataset,ImgLoader
import pandas as pd

cmd = "python train_surrogate.py ./logs/run-20250517_015519-test_data-67PT/media/wm_img/ ./logs/run-20250517_015519-test_data-67PT/media/no_wm_img/ ./models/ test_surr --mode=rawpix --vae=none --apply_fft --batch_size=4 --v"

cmd2 = "python remove_watermark.py ./logs/run-20250517_015519-test_data-67PT/media/wm_img/ ./models/test_surr.pth ./outputs/images/ --batch_size=1 --init_steps=1 --n_steps=20 --eps=5 --vae=none --apply_fft"

cmd3 = "python assess_images.py --run_name=test_run \
                                --original_images_path=./logs/run-20250517_015519-test_data-67PT/media/wm_img/ \
                                --adv_images_path=./logs/run-20250517_015519-test_data-67PT/media/no_wm_img/ \
                                --table_path=./logs/run-20250517_015519-test_data-67PT/media/table/metadata.csv \
                                --imagenet_path=./imagenet/ \
                                --watermark_path=./tr_params.pth \
                                --model_id=512x512_diffusion"

og_dir = "./logs/run-20250517_015519-test_data-67PT/media/wm_img/"
tab_dir = "./logs/run-20250517_015519-test_data-67PT/media/table/metadata.csv"
tab = pd.read_csv(tab_dir)
data = ImgLoader(og_dir,tab,key='wm_img')
print(data[0])
print(data[1])
print(data[2])