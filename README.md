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

- 对meta_json的处理在labels_res_cleaning.py中，可以根据需要修改。

- 常用工具都在`utils.py`中，可以根据需要修改。可能需要增加可视化工具。推荐mmcv或者直接用qwen的tokenizer的draw_bbox_on_latest_picture函数

- gpt4v生成assess和description的文件后面要加上对手动降质数据格式的适配，尝试使用gpt4o进行标注。

- The important results will be mainly saved in the `./results` directory.  
