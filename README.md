# MultiSocial: Multilingual Benchmark of Machine-Generated Text Detection of Social-Media Texts
Source code for replication of the experiments in the paper submitted to ACL Rolling Review.

## Install Dependencies
Install the [IMGTB framework](https://github.com/kinit-sk/IMGTB) along with its dependencies.

## Data Preparation
Download the human texts from the existing datasets, described in the paper.
- [Gab](https://zenodo.org/records/1418347)
- [WhatsApp](https://github.com/gvrkiran/whatsapp-public-groups)
- [Telegram](https://zenodo.org/records/3607497)
- [Twitter](https://gitlab.com/checkthat_lab/clef2022-checkthat-lab/clef2022-checkthat-lab/-/tree/main/task1)
- [Twitter](https://www.kaggle.com/datasets/kazanova/sentiment140/data)
- [Discord](https://www.kaggle.com/datasets/jef1056/discord-data)

## Source Code Structure
| # | Description |
| :-: | :-: |
| 01 | A script for combination of the downloaded datasets and selection of the subset. |
| 02a | A script to generate texts using paraphrasing by various large language models (HuggingFace pretrained models as well as OpenAI API accessible models). |
| 02b | A script for automated text-similarity metrics calculation. |
| 02c | A script for combination of the generated texts from different models. |
| 02d | A script for pre-processing (filtering) for the experiments. |
| 03a | A configuration file to run external [IMGTB framework](https://github.com/kinit-sk/IMGTB). |
| 03b | A text file to update environment requirements for fine-tuning process. |
| 03c | A script for fine-tuning of detection models. |
| 04 | Google Colab notebook for results analysis. |
