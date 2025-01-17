## Download models
Call `bash CNNDetection/weights/download_weights.sh` to download models into `CNNDetection/weights` folder. These weights are used in the pretrained models

# Demo runs
`CNNDetection` contains two demo runs. One takes the path to a single image, and returns the probability of this image being AI generated. The second takes the path to a folder, the folder should be structed with exactly two subfolders named `0_real` (containing real images) and `1_fake` (containg AI-generated images). After calling this program the model classifies all images in each folder, and displays its total accuracy across real and fake images, its accuracy on fake images and its accuracy on real images.

## Run tests with CCNDetection
The tests have been added to `tasks.py` as `cnn_detect` and `cnn_detect_dir` respectively. Run these from the main project folder with `invoke cnn-detect` or  `invoke cnn-detect`.

Examples
Running
`lucas@Surphase:~/MLOps_Project$ invoke cnn-detect` displays:
`probability of being synthetic: 0.00%`

And Running
`lucas@Surphase:~/MLOps_Project$ invoke cnn-detect-dir`
displays \
`Average sizes: [256.00+/-0.00] x [256.00+/-0.00] = [0.07+/-0.00 Mpix]`\
`Num reals: 2, Num fakes: 1`\
`AP: 50.00, Acc: 66.67, Acc (real): 50.00, Acc (fake): 100.00`

## TODO
Make these commands in `train.py` use our folder structure.

## Run the python scripts directly
Use of CNNDetect on single image:
Use `python demo.py -f </path_to/image.png> -m <path_to/blur_jpg_prob0.5.pth>`

Use of CNNDetect on image folder
`python demo_dir.py -d </path_to_folder> -m <path_to/blur_jpg_prob0.5.pth>`

## Reference
[CNNDetection](https://github.com/PeterWang512/CNNDetection)
