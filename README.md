# GenPrompt: Generating Descriptive Image Quality Assessment Prompts


## Acknowledgement

This part of the repository is based on [DepictQA](https://github.com/XPixelGroup/DepictQA) . Thanks for this awesome work.

## The original paper

Official pytorch implementation of the papers: 

- DepictQA-Wild (DepictQA-v2): [paper](https://arxiv.org/abs/2405.18842), [project page](https://depictqa.github.io/depictqa-wild/). 

    Zhiyuan You, Jinjin Gu, Zheyuan Li, Xin Cai, Kaiwen Zhu, Chao Dong, Tianfan Xue, "Descriptive Image Quality Assessment in the Wild," arXiv preprint arXiv:2405.18842, 2024.

- DepictQA-v1: [paper](https://arxiv.org/abs/2312.08962), [project page](https://depictqa.github.io/depictqa-v1/). 

    Zhiyuan You, Zheyuan Li, Jinjin Gu, Zhenfei Yin, Tianfan Xue, Chao Dong, "Depicting beyond scores: Advancing image quality assessment through multi-modal language models," ECCV, 2024.


## Update

📆 [2024.08] The first version of the GenPrompt repository is released.

## Installation

- Create environment. 

    ```
    # create environment
    conda create -n genprompt python=3.10
    conda activate genprompt
    pip install -r requirements.txt
    ```

## Datasets

- The example image datasets are provided [here](https://modelscope.cn/datasets/Siguax23/Quality_LLM_public_ORIGIN/files), which contains 77000 images in four zip files.

- The example annotation file is ./example_data/主客观筛选三轮完全置信小项.json, which needs to be converted to the format for training and evaluating as follows:

    ```

    ```


## Scripts

- cd the scripts directory: `cd ./scripts`. 

- You should run the scripts according to the sequence as follows. 

    | Script | Description |
    | -------- | -------- |
    |`./scripts/combine_json.py`| combine many meta jsons to one|
    | `./scripts/split_json.py` | split the example annotation file into one image per file |
    | `./scripts/extract_json.py` | find the corresponding image from example image datasets for each file |
    | `./scripts/convert_json.py` | combine the each distortion and bounding box information into distortion class|
    |`./scripts/distortion_dist.py`| draw the distribution of distortion types|
    |`./scripts/draw_bbox.py`| draw the bounding box and corresponding distortion type on the image|
    |`./scripts/bbox_dist.py`| draw the distribution of bounding box sizes|
    |`./scripts/gen_resp_desp.sh`|generate the image description from GPT-4  |
    |`./scripts/gen_resp_assess.sh`|generate the image assessment from GPT-4  |
    | `./scripts/image_distribution.py` | draw the distribution of image sizes|
    | `./scripts/train_and_val.py` | split prompts into training and validation sets |
    |`./scripts/check.py`| check whether the images correspond to train and val sets|
    | `./scripts/gen_gpt_score.sh` | GPT-4 score of quality accessment and reasoning tasks. You should generate the answer from the finetuned model before running this script. |
    |`./scripts/draw_score.py`| draw the distribution of gpt4-score|

- The important results will be mainly saved in the `./results` directory.  
