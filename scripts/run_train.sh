ENV=pixel_space/server_train_pixel poetry run python -m rectified_flow.training.train_unet_pixel_space
ENV=pixel_space/server_train_pixel_flickr poetry run python -m rectified_flow.training.train_unet_pixel_space_flickr30k
ENV=text_cond/server_train poetry run python -m rectified_flow.training.train_unet_pixel_flick_text_cond_film
ENV=live/pixel_space/server_train_pixel poetry run python -m rectified_flow.live.training.train_unet_pixel_space_live

# Unet++
ENV=server_train_pixel_flickr poetry run python -m rectified_flow.training.train_unet_pixel_space_flickr30k_01

# Unet++ w TC
ENV=server poetry run python -m rectified_flow.training.train_unet_pp_pixel_flickr_tc

# Unet++ w TC using CLIP
ENV=server poetry run python -m rectified_flow.training.train_unet_pp_pixel_flickr_tc_clip
ENV=server_test poetry run python -m rectified_flow.training.train_unet_pp_pixel_flickr_tc_clip

# Unet++ w TC using CLIP -> COCO
ENV=server poetry run python -m rectified_flow.training.train_unet_pp_pixel_coco_tc_clip

# Unet joint image+text
ENV=server poetry run python -m rectified_flow.training.train_unet_pixel_space_joint