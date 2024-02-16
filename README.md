## Overview

This project introduces a user-friendly end-to-end pipline for personalized video content creation. The framework is organized into three key
steps: Text-to-Speech, Voice Conversion, and Lip-Sync. The pipeline uses open-source models from the following projects:

- Text-to-Speech: [Bark](https://github.com/suno-ai/bark)

- Voice Conversion: [so-vits-svc](https://github.com/svc-develop-team/so-vits-svc), [so-vits-svc-fork](https://github.com/voicepaw/so-vits-svc-fork). In addition to this project, the pipeline employs a [noise removal](https://pypi.org/project/noisereduce/) to improve the sound quality of voice conversion output.

- Lip-sync: [video-retalking](https://github.com/opentalker/video-retalking)

# Demo

Example text: "A large language model is a deep learning algorithm that can perform a variety of natural language processing tasks. Large language models use transformer models and are trained using massive datasets. This enables them to recognize, translate, predict, or generate text or other content."

https://github.com/pn-pham/deepfakes_all_in_1/assets/130674444/1c2480a3-4417-4062-965b-9e2c080ea192

## Installation

1. Install [Miniconda](https://docs.anaconda.com/free/miniconda/) and Python.

2. Create a Conda environment and install the requirements
```bash
  git clone https://github.com/pn-pham/deepfakes_all_in_1.git

  cd deepfakes_all_in_1 

  conda create -n deepfakes python==3.11.5

  conda activate deepfakes

  pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2+cu118 --index-url https://download.pytorch.org/whl/cu118

  pip install -r requirements.txt
```

3. Install [espeak-ng](https://pypi.org/project/espeakng/) to enable text-to-speech synthesizer.
   
```bash
  pip install espeakng
```

If the installation with ```pip``` doesn't work, please use ```sudo apt-get```.

```
  sudo apt-get install espeak-ng
```

4. Download [pre-trained models](https://drive.google.com/drive/folders/18rhjMpxK8LVVxf7PI6XwOidt8Vouv_H0?usp=share_link), then put them in `./lip_sync/video_retalking/checkpoints`.
## Train voice conversion model

1. Prepare voice recordings of a target voice with any speech content. The total length of recordings is recommended at least 20 minutes to achieve an acceptable performance.

2. Place the recordings in folder ./data/vc_train/input_audio.

3. Run the command below to pre-process the data.
```bash
  python deepfakes.py vc-pre-processing
```
4. Run the command below to train a voice conversion model. It's recommended to train the model for 5000 epochs (or more). Please be aware that, this may take a day to complete.
```bash
  python deepfakes.py vc-train --max-epoch 5000
```

## Inference
### 1. Text-to-speech
1. Prepapare one (or some) script (.txt) for creating deepfake videos. Each file will be used to create one video.

2. Place the scripts in folder ```./data/input/text```.

3. Run command:
```bash
python deepfakes.py tts-bark
```

  If only one script is used, run command:
```bash
python deepfakes.py tts-bark --input-path ./path-to-file/audio.wav
```

The output audio will be store in folder ./data/tts.

### 2. Voice conversion

Run this command to apply voice conversion to all of the output audio:

```bash
python deepfakes.py vc-infer
```
  If only one audio needs to be converted, run this command:
```bash
python deepfakes.py vc-infer --input-path ./path-to-file/audio.wav
```
  Note that, the latest trained model in folder ./data/vc_train/model is used by default. To use a specific trained model (e.g. G_5000.pth), please include ```--model-path``` as follows:
```bash
python deepfakes.py vc-infer --model-path ./data/vc_train/model/G_5000.pth
```

### 3. Lip-sync

1. Prepare one (or some) videos for lip-sync. It is recommended that the videos are without lip movement.

2. Place them in folder ```./data/input/video```

3. Run the following command:

```bash
python deepfakes.py lip-sync
```

  The above command will apply lip-sync for every audio with a random video in the folder. To specify the audio and video, please use this command:

```bash
python deepfakes.py lip-sync --audio-path ./data/vc/converted_audio.wav --video-path ./data/input/video/video.mp4
```

   To make the output videos look different, the model starts the lip-sync from a random frame of the input video till the end. To specify the starting frame, please add the argument `--start-frame`:
   
```bash
python deepfakes.py lip-sync --start-frame 10 
```

  The ouput videos are store in folder `./data/lip_sync`
## Disclaimer

This project is created to use deepfakes for good. Please use this only for good. The authors do not take responsibility for any generated output.

## All Thanks To Our Contributors 

<a href="https://github.com/pn-pham/deepfakes_all_in_1/contributors">
  <img src="https://contrib.rocks/image?repo=pn-pham/deepfakes_all_in_1" />
</a>
