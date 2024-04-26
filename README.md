# API-BLEND
**A Comprehensive Corpora for Training and Benchmarking API LLMs.**

Paper Link: https://arxiv.org/abs/2402.15491


API-BLEND is a collection of 10 datasets for training and systematic testing of tool-augmented LLMs. The datasets mimic real-world scenarios involving API-tasks such as API / tool detection, slot filling, and sequencing of the detected APIs. Out of 10 datasets we have curated 6 datasets from the existing datasets, and the other 4, we have used them off-the-shelf (for OOD tests).

**Note:** Currently, we are in the process of obtaining license clearance to release the curated datasets directly. So, for the time, we have outlined the steps involved in curating them from the raw datasets. 
## Install Dependencies

```commandline
conda create --name api-blend python=3.9
conda activate api-blend
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

For LLM-based data-generation we have used [IBM Generative AI Python SDK](https://pypi.org/project/ibm-generative-ai/). Please follow the instructions to generate unique API key to access IBM Generative AI.
```commandline
pip install --upgrade ibm-generative-ai==2.2.0
```

## Datasets Curation
### 1. LLM-based Data Generation
- ##### Raw Data: 
    We have curated SeqSGD (from SGD) and SeqMultiWOZ (from MultiWOZ) using LLM. Please download the raw data from the following links.
      
    - SGD: [Link](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue)
    - MultiWOZ: [Link](https://github.com/budzianowski/multiwoz/tree/master/data/MultiWOZ_2.2)
  
- ##### Generate Data:
    Here is an example to generate SeqMultiWOZ, where `data/raw/MultiWOZ_2.2` is the raw data dir. Same codebase works for SeqSGD.
```commandline
export GENAI_API=<your API url>
export GENAI_KEY=<your API key>

python llm-based-generation/llm-data-gen.py \
	--data_dir data/raw/MultiWOZ_2.2 \
	--save_dir data/processed/SeqMultiWOZ \
	--dataset_name multiwoz \
	--model google/flan-t5-xxl
```

### 2. Grammar-based Data Generation
- ##### Raw Data:
  Using grammar based generation, we have generated 4 datasets. Please download the raw datasets from the following links 
  - ATIS: [Link](https://www.kaggle.com/datasets/hassanamin/atis-airlinetravelinformationsystem)
  - SNIPS: [Link](https://github.com/sonos/nlu-benchmark)
  - TopV2: [Link](https://fb.me/TOPv2Dataset)
  - ToolQA: [Link](https://github.com/night-chen/ToolQA)
  
- ##### Generate Data:
  
  - **SeqATIS and SeqSNIPS**:
        Please download the SNIPS and ATIS datasets from the above link. Run the below script to generate SeqSNIPS and SeqATIS datasets, where `data/raw/` contains the raw `SNIPS` and `ATIS` datasets.
  ```commandline
  python grammar-based-generation/SeqSNIPS_SeqATIS-data-gen.py \
      --data_dir data/raw/ \
      --save_dir data/processed/
  ```  
  - **SeqTopV2**:
        Please download the TopV2 following the above link. Run the below script to generate SeqTopV2 dataset, where `data/raw/TOPv2_Dataset` is the raw data.  
        
  ```commandline
  python grammar-based-generation/SeqTopV2-data-gen.py \
      --data_dir data/raw/TOPv2_Dataset \
      --save_dir data/processed/SeqTopV2
  ```     
