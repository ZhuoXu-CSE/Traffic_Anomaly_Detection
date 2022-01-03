# Traffic_Anomaly_Detection
Detecte anomalies based on video feeds available from multiple cameras at intersections and along highways. 
- Linux (tested on RedHat 8.2)
- Python 3.7
- PyTorch 1.10
- Opencv
- CUDA 11.3
- sklearn
- [mmcv](https://github.com/open-mmlab/mmcv)
- [mmdetection](https://github.com/open-mmlab/mmdetection)
- 
### Installation

1. Install PyTorch 1.10.1 and torchvision following the [official instructions](https://pytorch.org/).
2. Install [mmdetection@(pytorch1.10.1)](https://mmdetection.readthedocs.io/en/latest/get_started.html#installation) and mmcv (https://mmcv.readthedocs.io/en/latest/get_started/installation.html).
3. Train the model from scarch can be time consuming. Pre-trained weights and intermediate results  are available [here](https://drive.google.com/drive/folders/12gyMVO3JxGDvpknqAeoaevrvvtXUgHz7?usp=sharing)

## Run
- Download Models.zip and put its content into  `./models` and put the the other three files in `./detection_results`.
- Run `python ./src/bg_modeling/capture_and_average.py`.Upon its completion, original frames from the videos are saved as `./data/ori_images.hdf5` and superimposed frames are saved as`./data/processed_images.hdf5`  
- Run anomely detection on specific video: `python ./detect_anomaly.py <video id>`
- Run anomely detection on all video: `sh detect_all_videos.sh`
