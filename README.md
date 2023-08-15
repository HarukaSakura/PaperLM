# PaperLM
Source code for the CIKM 2023 paper "PaperLM: A Pre-trained Model for Hierarchical Examination Paper Representation Learning" 


## Environment

Requirments

```
numpy
pandas
torch==1.11.0
transformers==4.26.0
edunlp==0.0.8
tqdm
sklearn
scipy
```


## Usage

* Data preprocessing    
  * Convert paper text into vector using pre-trained BERT
  * Build knowledge table

  ```bash
  cd src
  # train
  python data_preprocess.py
  ```
  
* Pre-train

  ```bash
  cd src
  # train
  python main.py --mode pretrain
  ```

* Test

  ```bash
  cd src
  
  # paper difficulty estimation
  python main.py --mode finetune --downstream_task diff
  
  # examination paper retrieval
  python main.py --mode finetune --downstream_task similarity
  
  # paper clustering
  python main.py --mode finetune --downstream_task cluster
  ```

For more running arguments, please refer to [src/utils.py].
