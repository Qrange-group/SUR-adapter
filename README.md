# SUR-adapter 
![GitHub](https://img.shields.io/github/license/gbup-group/DIANet.svg)
![GitHub](https://img.shields.io/badge/Qrange%20-group-orange)

By [Shanshan Zhong](https://github.com/zhongshsh) and [Zhongzhan Huang](https://dedekinds.github.io) and [Wushao Wen](https://scholar.google.com/citations?user=FSnLWy4AAAAJ) and [Jinghui Qin](https://github.com/QinJinghui) and [Liang Lin](http://www.linliang.net)

This repository is the implementation of "SUR-adapter: Enhancing Text-to-Image Pre-trained Diffusion Models with Large Language Models" [[paper]](https://arxiv.org/abs/2305.05189). 


## 🌻 Introduction

**Semantic Understanding and Reasoning** adapter (SUR-adapter) for pre-trained **diffusion models** can acquire the powerful semantic understanding and reasoning capabilities from **large language models** to build a high-quality textual semantic representation for text-to-image generation. 

<p align="center">
  <img src="https://github.com/Qrange-group/RAS/assets/62104945/af863827-2ea4-45cb-b3ed-2f98ba0e7d03">
</p>

## 📣 News


2023/10/20 - We have provided an example checkpoint of SUR-adapter [[Google Drive](https://drive.google.com/drive/folders/1UyC9_AqTezmHXmj4dh0A-9RBKKx_JmJZ?usp=share_link)]. Please try it! 

2023/08/19 - We have provided the data scraping code for Civitai. Please take a look at [processing](https://github.com/Qrange-group/SUR-adapter/blob/main/data_collect/processing.ipynb).

## 🏇 TODO

- [x] data collection script
- [x] pretrain model
- [ ] dataset

## 🌻 Quick Training

We only provide **ONE** data sample for running through this `repo` with the following code. See dataset declaration for details.

(1) Clone the code. 

```sh
git clone https://github.com/Qrange-group/SUR-adapter
```
```sh
cd SUR-adapter
```

(2) Prepare the enviroment.

If **Pytorch** is not installed, you can install it through the [official website guide](https://pytorch.org/get-started/locally). For example, when I use `nvidia-smi` to know that my `CUDA Version` is 11.1, we can install Pytorch through the following command:
```sh
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu111
```

Then install `diffusers` following the [guide](https://huggingface.co/docs/diffusers/installation).
```sh
pip install diffusers["torch"]
```

Finally, install the relevant packages.
```sh
pip install -r requirements.txt
```

(3) Run the following code in shell, where `0` is the gpu id. If you encounter CUDA out of memory, you can try to find a solution in [document](https://huggingface.co/docs/diffusers/v0.16.0/en/optimization/fp16). 

```sh
sh run.sh 0
```

**Quick Training** only uses about 5200 MiB GPU Memory. If your GPU memory is large enough, you can increase the batch size or not use mixed precision. The following is a description of the parameters of `run.sh`, the details can be found in `SUR_adapter_train.py`. 

```sh
export CUDA=$1               # GPU id 
export LLM="13B"             # size of LLM
export LLM_LAYER=39          # layer of LLM
export MODEL_NAME="runwayml/stable-diffusion-v1-5"  # pre-trained diffusion model
export INFO="test"           # help to idetify the checkpoints
export OUTPUT_DIR="fp16"     # help to idetify the checkpoints
export TRAIN_DIR="sur_data_small"   # dataset
export SAVE_STEP=100         # step saved at intervals
export BATCH_SIZE=1          # batch size

# please see https://huggingface.co/docs/diffusers/v0.16.0/en/training/text2image to get more details of training args
CUDA_VISIBLE_DEVICES=$CUDA accelerate launch SUR_adapter_train.py \    
  --mixed_precision="fp16" \
  --info=$INFO \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$TRAIN_DIR \
  --output_dir=$OUTPUT_DIR \
  --llm=$LLM \
  --llm_layer=$LLM_LAYER \
  --checkpointing_steps=$SAVE_STEP \
  --train_batch_size=$BATCH_SIZE \
  --resolution=512 --center_crop --random_flip \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=5000 \
  --learning_rate=1e-05 \
  --prompt_weight=1e-05 \
  --llm_weight=1e-05 \
  --adapter_weight=1e-01 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 
```

## 🌻 Dataset Declaration

You can prepare the dataset in the format of `sur_data_small`. For examples, you can get data from [Civitai](https://civitai.com) through [api](https://github.com/civitai/civitai/wiki/REST-API-Reference). We have provided the data scraping code for Civitai. Please take a look at [processing](https://github.com/Qrange-group/SUR-adapter/blob/main/data_collect/processing.ipynb). If you have some problems, you can try to find answers from [datasets document](https://huggingface.co/docs/datasets/create_dataset) for more details. 

❣ **Warning** ❣: The dataset SURD proposed in our work is collected from [Lexica](https://lexica.art) ([license](https://lexica.art/license)), [Civitai](https://civitai.com) ([license](https://github.com/civitai/civitai/blob/main/LICENSE)), and [Stable Diffusion Online](https://stablediffusionweb.com) ([license](https://huggingface.co/spaces/CompVis/stable-diffusion-license)). The licenses point out that if the dataset is used for commercial purposes, there may be certain legal risks. In order to avoid potential copyright disputes and unnecessary trouble, we have decided not to publicly release SURD and our ckpt. If you have a need for the data, please collect and clean it yourself. If it is to be used for commercial purposes, please contact the relevant website or author for authorization.

 

## 🌻 Prompt2vec

We utilize [LLaMA](https://github.com/facebookresearch/llama), a collection of foundation language models ranging from 7B to 65B parameters, as knowledge distillation for large language models (LLMs). Specifically, we save the vector representations of simple prompts in `i`-th layer of LLMs, which serve as the text understanding to finetune diffusion models. If you want to output the vectors from [LLaMA](https://github.com/facebookresearch/llama), we recommend that you can focus on [following two lines](https://github.com/facebookresearch/llama/blob/main/llama/model.py#L234-L235) of [LLaMA](https://github.com/facebookresearch/llama).

```python
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
```

The data format for prompt2vec is as follows. 

```
{
  "prompt": torch.tensor,
}
```

When you are ready for prompt2vec's `.pt` type file, please save the `.pt` file to the prompt2vec folder. For example, you can save the prompt vectors from the fortieth layers of LLaMA (13B) to `prompt2vec/13B/39.pt`. 

## 🌻 Inference

Run the `demo.ipynb`.

```python
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

from SUR_adapter_pipeline import SURStableDiffusionPipeline
import torch
from SUR_adapter import Adapter

adapter_path = "checkpoints/runwayml_fp16/test_llm13B_llml39_lr1e-05_llmw1e-05_promptw1e-05_adapterw0.1/adapter_checkpoint1000.pt"
adapter=Adapter().to("cuda")
adapter.load_state_dict(torch.load(adapter_path))
adapter.adapter_weight = float(adapter_path.split("adapterw")[-1].split('/')[0])

model_path = "runwayml/stable-diffusion-v1-5"
pipe = SURStableDiffusionPipeline.from_pretrained(model_path, adapter=adapter)
pipe.to("cuda")
pipe.safety_checker = lambda images, clip_input: (images, False)

image = pipe(prompt='An aristocratic maiden in medieval attire with a headdress of brilliant feathers').images[0]
image.show()
```

## 🌸 Citation

```
@article{zhong2023adapter,
  title={SUR-adapter: Enhancing Text-to-Image Pre-trained Diffusion Models with Large Language Models},
  author={Zhong, Shanshan and Huang, Zhongzhan and Wen, Wushao and Qin, Jinghui and Lin, Liang},
  journal={arXiv preprint arXiv:2305.05189},
  year={2023}
}
```

## 💖 Acknowledgments

Many thanks to [huggingface](https://github.com/huggingface) for their [diffusers](https://github.com/huggingface/diffusers) for image generation task. I love open source. 


