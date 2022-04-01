# Contrastive for Few-shot Language Learners

This repo covers the implementation of the following paper:  **"Contrastive Learning for Prompt-based Few-shot Language Learners"** .
<img src="figures/overview.png" width="800">

If you find this repo useful for your research, please consider citing the paper.

Our code is  heavily borrowed from [LM-BFF](https://github.com/princeton-nlp/LM-BFF) and [SupCon](https://github.com/HobbitLong/SupContrast) (```/src/losses.py```).

## Requirements

This repo was tested with Ubuntu 18.04.5 LTS, Python 3.7, PyTorch 1.6.0, and CUDA 10.1. You will need a 48 GB GPU for experiments with RoBERTa-base, and 4x 48 GB GPUs for RoBERTa-large. We run our experiments on Nvidia RTX-A6000 and RTX-8000, but Nvidia A100 with 40 GB should also work.

## Download data
We use pre-processed datasets (SST-2, SST-5, MR, CR, MPQA, Subj, TREC, CoLA, MNLI, SNLI, QNLI, RTE, MRPC, QQP) from  [LM-BFF](https://github.com/princeton-nlp/LM-BFF). LM-BFF offers helpful scripts for downloading and preparing the dataset. Simply run the commands below.
```shell
cd data
bash download_dataset.sh
```
Then use the following command to generate 16-shot datasets we used in the study.
```shell
python tools/generate_k_shot_data.py
```

## Running our fine-tuning
The primary prompts (templates) used for tasks have been pre-defined in ```run_experiments.sh```. The auxiliary templates used when generating multi-views of inputs for contrastive learning can be found in ```/auto_template/$TASK```.

Assuming you have one GPU in you system, we show an example of running our fine-tuning on SST-5 (random templates and random demonstrations for "augmented views" of inputs).

```shell
for seed in 13 21 42 87 100   #### random seeds for different train-test splits
do
    for bs in 40   #### batch size
    do
        for lr in 1e-5    #### learning rate for MLM loss
        do
            for supcon_lr in 1e-5    #### learning rate for SupCon loss
            do
                TAG=exp \
                TYPE=prompt-demo \
                TASK=sst-5 \
                BS=$bs \
                LR=$lr \
                SupCon_LR=$supcon_lr \
                SEED=$seed \
                MODEL=roberta-base \
                bash run_experiment.sh
            done
        done
    done
done

rm -rf result/
```
Our framework also applies to prompt-based method without demonstrations, i.e., ```TYPE=prompt``` (In this case, we only randomly sample templates for generating "augmented views"). The results are saved in ```log```.



Using RoBERTa-large as the base model requires 4 GPUs, each with 48 GB of memory. You need to first edit Line 20 in ```src/models.py``` to be ```def __init__(self, hidden_size=1024)```.

```shell
for seed in 13 21 42 87 100   #### random seeds for different train-test splits
do
    for bs in 10   #### batch size for each GPU, total batch size is then 40
    do
        for lr in 1e-5    #### learning rate for MLM loss
        do
            for supcon_lr in 1e-5    #### learning rate for SupCon loss
            do
                TAG=exp \
                TYPE=prompt-demo \
                TASK=sst-5 \
                BS=$bs \
                LR=$lr \
                SupCon_LR=$supcon_lr \
                SEED=$seed \
                MODEL=roberta-large \
                bash run_experiment.sh
            done
        done
    done
done

rm -rf result/
```



## Collecting results
```
python tools/gather_result.py --condition "{'tag': 'exp', 'task_name': 'sst-5', 'few_shot_type': 'prompt-demo'}"
```
It will collect the results from ```log``` and compute the mean and standard deviation over those 5 train-test splits.

## Contacts
For any questions, please contact authors.


## Acknowlegements
Thanks to [LM-BFF](https://github.com/princeton-nlp/LM-BFF) and [SupCon](https://github.com/HobbitLong/SupContrast), for the preliminary implementations.
