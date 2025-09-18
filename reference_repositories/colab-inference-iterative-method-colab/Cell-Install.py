import base64
#@markdown # Install

%cd /content
!git clone -b colab-inference-iterative-method https://github.com/Qupci/Music-Source-Separation-Training
!git clone https://github.com/Qupci/MVSep-API-Examples

#requirements fix by santilli_
req_text = """
mutagen==1.47.0
ml_collections==1.1.0
numpy>=1.26.0
pandas==2.2.2
scipy
tqdm
segmentation_models_pytorch==0.3.3
timm
audiomentations
pedalboard
omegaconf
beartype
rotary_embedding_torch==0.3.5
einops
# librosa==0.11.0
demucs #==4.0.0
# transformers==4.35.0
torchmetrics==0.11.4
spafe==0.3.2
protobuf
torch_audiomentations
asteroid==0.7.0
auraloss
torchseg
"""

with open("Music-Source-Separation-Training/requirements.txt", "w") as f:
    f.write(req_text)

!mkdir '/content/Music-Source-Separation-Training/ckpts'

print('Installing the dependencies... This will take a few minutes')
!pip install -r 'Music-Source-Separation-Training/requirements.txt' &> /dev/null

print('Installation is done !')