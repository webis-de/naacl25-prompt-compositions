# Adaptive Prompting: Ad-hoc Prompt Composition for Social Bias Detection

Code for the paper [Adaptive Prompting: Ad-hoc Prompt Composition for Social Bias Detection](https://arxiv.org/abs/2502.06487).

For details on the approach, architecture and idea, please see the published paper.

```
@inproceedings{spliethover-etal-2025-adaptive,
    title =      "Adaptive Prompting: Ad-hoc Prompt Composition for Social Bias Detection",
    author =     Splieth{\"o}ver, Maximilian and Knebler, Tim and Fumagalli, Fabian and Muschalik, Maximilian and Hammer, Barbara and H{\"u}llermeier, Eyke and Wachsmuth, Henning,
    booktitle =  "Proceedings of the 2025 Conference of the North American Chapter of the Association for Computational Linguistics",
    month =      apr,
    year =       "2025",
    address =    "Albuquerque, New Mexico",
    publisher =  "Association for Computational Linguistics",
    url =        "https://arxiv.org/abs/2502.06487",
}

```

---

_All experiments were conducted using Python 3.10.4_

## Install Dependencies

```bash
conda create --name prompt-compositions python=3.10.14 -y

conda activate prompt-compositions

# choose the appropriate CUDA version for your environment
conda install cuda -c nvidia/label/cuda-12.1.0

pip install -r requirements.txt

# we need to upgrade outlines to 0.0.39 in order to integrate with vLLM
pip install outlines==0.0.39 --force-reinstall --no-deps

# custom library for easier inference
pip install -e ./src/flex-infer
```

## Load the Models

To run the experiments, download the pre-trained LLMs from [Hugging Face](https://huggingface.co/):

- Mistral: <https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2>
- Command R: <https://huggingface.co/CohereForAI/c4ai-command-r-v01>
- Llama 3: <https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct>

You can set the path to the models using the ```models``` variable at the beginning of the ```run_experiments.sh``` script.

To obtain the similarity-based few-shot examples for the bias detection experiments, you’ll need to download the ```all-mpnet-base-v2``` embedding model. By default, the model is expected at ```models/all-mpnet-base-v2```, but you can customize this path by modifying the ```SENTENCE_TRANSFORMER_MODELS``` variable in the ```src/bias_detection/config/settings.py``` file.

The model is available on Hugging Face: <https://huggingface.co/sentence-transformers/all-mpnet-base-v2>

## Load the Datasets

The datasets can be found here:

- SBIC: <https://maartensap.com/social-bias-frames/>
- CobraFrame: <https://huggingface.co/datasets/cmu-lti/cobracorpus>
- Stereoset: <https://huggingface.co/datasets/McGill-NLP/stereoset>
- SemEval-2014-ABSA: <https://huggingface.co/datasets/FangornGuardian/semeval-2014-absa>

Save the datasets in the ```datasets/``` directory and specify their paths using the ```DATASET_PATHS``` variable in the ```src/bias_detection/config/settings.py``` file.

## Run Experiments

To run the social bias detection experiments, execute the ```run_experiments.sh``` script with the path to the Python script for each dataset. The Python scripts for each dataset can be found in the ```experiments/``` directory.

```bash
# path to the python script for the experiment: experiments/sbic_greedy.py
# prefix for the output files to distinguish the results from different runs: sbic-greedy_
# data split: test
# number of GPUs used: 1
# more arguments and their default values can be found in the script

./run_experiments.sh experiments/sbic_greedy.py sbic-greedy_ test 1
```

## Pre-trained models

The trained models and their predictions on all datasets evaluated in the paper can be found on Hugging Face:

- [Finetune baseline models](https://huggingface.co/webis/naacl25-prompt-compositions_finetune-baseline)
- [Social bias classification models](https://huggingface.co/webis/naacl25-prompt-compositions_composition-prediction)
