# Guideline-Based Medical Reasoning in LLMs

This repository contains the code for the paper "Guideline-Based Medical Reasoning in LLMs",
accepted at ML4H 2026 (Machine Learning for Health).

Paper: https://arxiv.org/abs/2512.03838

## Abstract

Large Language Models (LLMs) excel at medical question-answering, yet progress toward medical reasoning models is limited by the lack of scalable, trustworthy reasoning supervision. We propose tapping into the so far underused resource of medical consensus guidelines. These map numerical clinical measurements to disease scores and can optionally be curated for site-specific exceptions. Automatically generating gold-standard reasoning chains from consensus rules enables both large-scale supervised fine-tuning and automatic evaluation. Applying SOFA, SAPS-II, and Sepsis-3 guidelines to MIMIC-III, we show that small fine-tuned models significantly outperform larger one-shot prompted models in derivation correctness and F1 score, achieving near-perfect deductive inference and reliable exception handling. Trustworthy step-by-step reasoning does not reliably emerge from one-shot prompting, even when models receive the consensus definition and an example. Performance across all models declines when forecasting future clinical variables, indicating that, on the evaluated structured ICU tasks, the central challenge is not rule application, but the orthogonal problem of generalizing into the future for prognostic reasoning.

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{staniek2026guideline,
  title     = {Guideline-Based Medical Reasoning in {LLMs}},
  author    = {Staniek, Michael and Sokolov, Artem and Riezler, Stefan},
  booktitle = {Proceedings of the Sixth Machine Learning for Health Symposium (ML4H)},
  series    = {Proceedings of Machine Learning Research},
  volume    = {346},
  publisher = {PMLR},
  address   = {Sydney, Australia},
  year      = {2026},
  month     = dec,
  url       = {https://arxiv.org/abs/2512.03838}
}
```

## Installation

We installed the environment like this:

```bash
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
source miniforge3/bin/activate
mamba create -n flamingo2 python=3.10.*
conda activate flamingo2
mamba install nvidia/label/cuda-12.4.1::cuda
mamba install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.4 accelerate -c pytorch -c nvidia
pip install flash-attn --no-build-isolation
pip install transformers datasets evaluate tqdm accelerate peft sacrebleu matplotlib scikit-learn
```

## Data Preprocessing

All commands below are meant to be run from the repository root, with the root on the Python path
(needed so that scripts in `pretrain/` and `testing/` can import `models/`, `layers/` and `utils/`):

```bash
export PYTHONPATH=$(pwd)
```

Download MIMIC-III from here: https://physionet.org/content/mimiciii/1.4/

Then run:

```bash
python preprocessing/preprocess1_mimic_iii_events_icu.py
python preprocessing/preprocess2_events_icu_pickle.py
python preprocessing/preprocess3_limitforward.py
python preprocessing/preprocess4_icd_codes.py
python preprocessing/preprocess5_icd_codes_ood.py
python preprocessing/preprocess6_real_exceptions.py
python preprocessing/preprocess8_sapsii.py
```

## Model Training

```bash
python pretrain/pretrain_LLM_textual_reduced.py
python pretrain/pretrain_LLM_textual_reduced_pipeline.py
python pretrain/pretrain_LLM_textual_reduced_multimodal.py
python pretrain/pretrain_LLM_textual_reduced_icd.py
python pretrain/pretrain_LLM_textual_reduced_real_exceptions.py
python pretrain/pretrain_LLM_textual_reduced_sapsii.py
```

## Testing

```bash
python testing/test_LLM_textual_reduced.py --load_path train_path
python testing/test_LLM_textual_reduced_pipeline.py --load_path train_path
python testing/test_LLM_textual_reduced_multimodal.py --load_path train_path
python testing/test_LLM_textual_reduced_icd.py --load_path train_path
python testing/test_LLM_textual_reduced_icd_ext.py --load_path train_path
python testing/test_LLM_textual_reduced_real_exceptions.py --load_path train_path
python testing/test_LLM_textual_reduced_sapsii.py --load_path train_path
```

## Prompting

```bash
python testing/test_LLM_textual_reduced_scratch_1shot_explanation_deepseek.py
python testing/test_LLM_textual_reduced_scratch_1shot_explanation_llama8b.py
python testing/test_LLM_textual_reduced_scratch_1shot_explanation_medllama.py

python testing/test_LLM_textual_reduced_scratch_1shot_explanation_real_exceptions_deepseek.py
python testing/test_LLM_textual_reduced_scratch_1shot_explanation_real_exceptions_llama8b.py
python testing/test_LLM_textual_reduced_scratch_1shot_explanation_real_exceptions_medllama.py

python testing/test_LLM_textual_reduced_scratch_1shot_explanation_sapsii_deepseek.py
python testing/test_LLM_textual_reduced_scratch_1shot_explanation_sapsii_llama8b.py
python testing/test_LLM_textual_reduced_scratch_1shot_explanation_sapsii_medllama.py
```

## Value Correctness and Inference Correctness

Choose the script matching your setting and run:

```bash
python analyze/analyse_results_latex.py generations_file
python analyze/analyze_results_sapsii.py generations_file
python analyze/analyse_results_latex_fake_preexisting.py generations_file
python analyze/analyse_results_latex_real_preexisting.py generations_file
```

