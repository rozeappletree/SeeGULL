***Technical setup:*** What are the key things you try to quantify in this study and how do you define and measure them? Give the key technical details: what models you use, datasets, prompts, the metrics used.

> Example things you might try to quantify: deception, faithful of CoT, model confidence, model confusion
---

⚠️ Handwritten

---
### A. Inference providers, and local models, GPUs used
---

1. minirouter.sh
   
   <img width="230" height="113" alt="image" src="https://github.com/user-attachments/assets/25a23a18-99b4-4301-99fa-ac82666c381d" />

2. opusgate.dev

   <img width="293" height="299" alt="image" src="https://github.com/user-attachments/assets/d477a36f-2b19-4030-8160-f9f8dfff754b" />

3. LLaMa 2 Chat HF (from `NousResearch` org instead of `meta-llama` because it needed proper auth)

4. Nvidia L40S from clore.ai, 40GB VRAM

5. Total budget ~ 150 USD worth Crypto



---
### B. Data, & Evals
---

Data used for training and evaluation can be divided into following three parts mentioned 

#### 1. Existing datasets used

- TruthfulQA dataset
  - It was not used directly anywhere in the training
  - It was _indirectly_ used in evaluation (see objective truth matrix)
  - source: HF data hub [link to file in project](https://github.com/rozeappletree/mats12/blob/1baae17e373dffc4c94b93ca43656eaf9bb5a6bd/data/truthfulqa/truthful_qa.json)
  - [Comprehensive EDA notebook](https://github.com/rozeappletree/easy-prey/blob/main/nb/TruthQA.ipynb)

#### 2. Synthetic datasets created using existing datasets

- **TrueGULL Conversations:** Conversational data built on top of TruthfulQA dataset
  * 2 classes: Gullible vs. non gullible human assistant conversations on TruthfulQA claims (class=high/low gullibility)
    > Well class-balanced dataset
  * data is divided into 2 parts, out of which, the second part is again divided into 2 more parts
    * **TrueGULL Defence 484** (pt. 1 of 2)
      * Generated using complement of "hard negatives" i.e 484 non-hard negatives of objective truth matrix
        * 6 conversations per single claim - 3 with high gullibility traits, 3 with low gullibility traits i.e `484x6` conversation files (note that some qns did not generate due to network / api issues)
      * script used for generating the dataset: [gen_defense_data484.py](https://github.com/rozeappletree/mats12/blob/1baae17e373dffc4c94b93ca43656eaf9bb5a6bd/scripts/gen_defense_data484.py)
        * **Recommended:** read the top docstring for entire _process, model used, prompts, etc._
      * dataset file: [datasets_defense_484.zip](https://github.com/rozeappletree/mats12/blob/1baae17e373dffc4c94b93ca43656eaf9bb5a6bd/datasets_defense_484.zip)
    * **TrueGULL Hard 333** (pt. 2 of 2)
      * "Hard negatives" of objective truth matrix.
      * `333x6` samples minus the corrupted ones
      * Same steps as above but for hard negative claims
        * `alibaba/qwen3.7-plus` (served through minirouter.sh) used instead of `GPT-5.6-Sol` (served through opusgate.dev) above
          * at the time, opus model was causing 500 internal server error at opusgate
      * **Recommended:** read the top docstring and prompt of the script used for generating the dataset: [gen_hard_data333.py](https://github.com/rozeappletree/mats12/blob/1baae17e373dffc4c94b93ca43656eaf9bb5a6bd/scripts/gen_hard_data333.py)
      * TrueGULL Hard 333 is divided into 2 parts:
        * **TrueGULL Ardulous 66:** Extremely Hard Subset (intersection of hard negative claims tested with certain methods) [`datasets_ardulous_66.zip`](https://github.com/rozeappletree/mats12/blob/1baae17e373dffc4c94b93ca43656eaf9bb5a6bd/datasets_ardulous_66.zip)
          * see: [`split_hard333`](https://github.com/rozeappletree/mats12/blob/1baae17e373dffc4c94b93ca43656eaf9bb5a6bd/scripts/split_hard333.py) (methods explained inside scripts and metadata)
        * **TrueGULL Just Hard 267:** [`datasets_justhard_267.zip`](https://github.com/rozeappletree/mats12/blob/1baae17e373dffc4c94b93ca43656eaf9bb5a6bd/datasets_justhard_267.zip) complement of ardulous 66
        * See: [`eda_hard_negatives_333.ipynb`](https://github.com/rozeappletree/mats12/blob/1baae17e373dffc4c94b93ca43656eaf9bb5a6bd/nb/eda_hard_negatives_333.ipynb) for comprehensive EDA
       
          * top 15 categories retained in hard 333
          
          <img width="1121" height="404" alt="image" src="https://github.com/user-attachments/assets/30b0a612-55be-4d29-b8d4-d6c12cf6e91d" />

- **LLama2 Conversations:** Similar gullible / non-gullible conversations were created locally in Nvidai L40S  (there are other 3 classes, mentioned in below 3.)
  * [datasets_llama2_deduplicated.zip](https://github.com/rozeappletree/mats12/blob/1baae17e373dffc4c94b93ca43656eaf9bb5a6bd/datasets_llama2_deduplicated.zip) raw data was originally having duplicates, removed them.
  * [gen_llama_dataset.py](https://github.com/rozeappletree/mats12/blob/1baae17e373dffc4c94b93ca43656eaf9bb5a6bd/scripts/gen_llama_dataset.py) (read doc string)
  
####  3. Novel synthetic datasets created

- **Human Behavioral Toy Dataset:** Synthetic data simulating four types of behaviours in conversational text
  * The four classess are: `rationality`, `seriousness`, `gullibility`, and `certainity seeking`
  * 3 classes - low, medium, high.
  * very small dataset- used local llama2chat, claude opus and gpt sol
  * Full EDA: [`eda_dataset_distributions_4way.ipynb`](https://github.com/rozeappletree/mats12/blob/main/nb/eda_dataset_distributions_4way.ipynb)
  * [`gen_opus_data100.py`](https://github.com/rozeappletree/mats12/blob/main/scripts/gen_opus_data100.py), [gen_sol_data100.py](https://github.com/rozeappletree/mats12/blob/main/scripts/gen_sol_data100.py) & above llama data
  * Showing distribution specifically for this to indicate how small this dataset is!
    
    <img width="899" height="338" alt="image" src="https://github.com/user-attachments/assets/ebeb4b4b-4e66-48ba-8106-24abeb7cd8e5" />

    <img width="1070" height="337" alt="image" src="https://github.com/user-attachments/assets/7ffda02c-b203-403a-81a1-b1f022eadf8e" />
    

   
- **PracGULL:** Conversational data for practical assistants
  * Given system prompt S, and human input H, we have two synthetic pairs:
    * P: Assistant response for gullible user
    * Q: Assistant response for non-gullible user
  * Built to automate steering analysis for specific usecase
  * *out of context for 20h window*, for more info see: https://github.com/rozeappletree/mats12/tree/main/data/PractGULL



---
### C. Models
---

### D. Prompts

### E. Eval

### F. Training Process

### 0.1

### 0.2

### 0.3

### 0.4

### G. Causality tests / Emprical study with webapp
