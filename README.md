# CLEVR Dataset Generation

This is the code used to generate the CLEVR video data.
1. git clone this repo
2. Download blender-2.78c and unzip it. The version must be 2.78c.
3. Add alias blender=/PATH_TO_BLENDER/blender to bashrc.
4. command line echo /PATH_TO_CODE/image_generation >> /PATH_TO_BLENDER/2.78/python/lib/python3.5/site-packages/clevr.pth
5. command line  blender --background --python /PATH_TO_CODE/image_generation/video_generate.py -- --num_images 1000 --use_gpu 1
6. The videos are saved as png files in the output folder.



I also write another 2 files video_generate_v2.py and bbox.py. The video_generate_v2.py can directly output videos instead of png image and it uses a built-in function in BLENDER that takes in key frames and output interpolations. The bbox.py will also produce bounding boxes of all objects.
