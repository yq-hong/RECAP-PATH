# RECAP-PATH (REasoning and Classification via Automated Prompting in PATHology images)

## Paper: Pathology Reasoning through Adaptive Multimodal-LLM Prompting

## Dataset
### BRACS:  BReAst Carcinoma Subtyping
#### Data Download
https://www.bracs.icar.cnr.it/download/

### BACH: BreAst Cancer Histology
#### Data Download
[https://www.bracs.icar.cnr.it/download/](https://iciar2018-challenge.grand-challenge.org/Dataset/)

### SICAPv2: Prostate Whole Slide Images with Gleason Grades Annotations
#### Data Download
[https://www.bracs.icar.cnr.it/download/](https://data.mendeley.com/datasets/9xxm58dvs3/1)

#### Preprocess
To make sure all images are less than 20MB to satisfy the OpenAI requirements.
For `datasets/BRACS/BRACS_RoI/latest_version/test/0_N`
```
python compress.py --set test --class_dir 0_N
```

Prepare the image paths for training:
```
python img_file_name.py --set test --class_dir 0_N
```

## Baselines
### Zero shot
```
python zero_shot.py --model gemini --task BRACS_multi --mode test --n_test 30 --out_num 1
```

## Phase-I: Optimizing Diversity
```
cd diversity
python main.py --task BRACS --model gemini --gradient_model gemini --out_num 1
```

## Phase-II: Optimizing Accuracy
```
cd prompt_optimization
python main_BRACS.py --reject_on_errors --task BRACS --diverse_init --diverse_exp 1 --class0 N --class1 IC --out_num 1
```

For usage instructions. Some of the arguments include:

* `--task`: Task name like 'BRACS', 'BRACS_multi', 'BACH', 'SICAPv2'.
* `--data_dir`: Directory where the task data file resides.
* `--diverse_init`: Whether to use diverse initial prompts from Phase I.
* `--diverse_exp`: Phase I output name.
* `--out`: Output name.
* `--max_threads`: Maximum number of threads to be used.
* `--val_score`: Whether to use validation set.
* `--class0`: class 0 Name (Ex: N, DCIS)
* `--class1`: class 1 Name (Ex: I, C, IC)

## After optimization evaluation
```
python classification.py --generate --task BRACS_multi --result_folder prompt_optimization --mode test --n_test 30 --exp 1 --prompt_idx 1 --out_num 1
```

* `--generate`: To generate new descriptions for images.
