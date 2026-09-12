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

- **GullibleConversations v0.3 & v1:** General conversations 

---
### C. Models
---

In the data section, click on the scripts to see models used.

The following models were used:

1. `NousResearch/llama2-chat-hf` primary model under investigation
2. Claude Opus 5 from opusgate.dev for generating synthetic data of human behaviours
3. GPT Sol from opusgate.dev for generating synthetic data
4. Deepseek V4 pro to generate bigger data size for v0.3 and v1 via minirouter.sh

---
### D. Prompts
---

Multiple prompts used, optimized for API calls and used loops for diverse outputs. See `B. Data & Evals`, click on scripts for prompts.

---
### F. Training & Evals
---

To be able to successfully prove that LLMs can detect human gullibility in human-assistant conversations, I am releasing reading probes that can detect human gullibility & control probes that can steer the assistant conversation based on gullibility level of the user.

- To verify if Chen et al's method can detect behavioral traits other than what was mentioned in their paper,
   - Started with 4 classes using "Human Behavioral Toy Dataset" to see the extendabiliy
   - Trained control and reading probes with the dataset, ckpts: [`probe_checkpoints.withLLaMaOpusSol.zip`](https://github.com/rozeappletree/mats12/blob/main/probe_checkpoints.withLLaMaOpusSol.zip)
   - Built a webapp to experimentally verify the usefulness of the model for both reading & steering the 4 class behaviors (seriousness, rationality, cetainity seeking and gullibiliy)
      ```
      # USAGE (4 CLASSES)
          python  webui/app.py  \
              --probe-dir /root/SeeGULL/mats12/probe_checkpoints.withLLaMaOpusSol/control_probe/llama2_sample2+claudeopus_sample2+sol_100 \
              --reading-probe-dir /root/SeeGULL/mats12/probe_checkpoints.withLLaMaOpusSol/reading_probe/llama2_sample2+claudeopus_sample2+sol_100/
      ```
   - Results were decent enough (see `G.` PDF in webapp section) and gave me confidence that moving forward in the direction will be fruitful
   - Filtered only the gullibility class samles [`datasets_regular_gullibility_170.zip`](https://github.com/rozeappletree/mats12/blob/main/datasets_regular_gullibility_170.zip) to train `v0.1`

<img width="1555" height="755" alt="image" src="https://github.com/user-attachments/assets/c9b27751-def0-4e4f-81a3-9d38294a5d37" />
<img width="1555" height="755" alt="image" src="https://github.com/user-attachments/assets/c1874327-79fb-4116-8170-e94df657d78a" />

<img width="1934" height="874" alt="image" src="https://github.com/user-attachments/assets/ef67b7ca-2ff8-4f2a-add4-ed69f2eb132a" />
<img width="1934" height="874" alt="image" src="https://github.com/user-attachments/assets/a1a45feb-439c-465a-81e5-56baeaf06d51" />

Note: all the plots and evals and confusion matrices are in ckpts folder

eg. confusion matrix for gullibility (of 4 classes)

<img width="300" height="300" alt="image" src="https://github.com/user-attachments/assets/16b9149d-0a34-4a06-874f-a90afbf1cceb" />

> old data has lesser samples

### v0.1

- Used only [`datasets_regular_gullibility_170.zip`](https://github.com/rozeappletree/mats12/blob/main/datasets_regular_gullibility_170.zip) to train reading and control probes
- Checkpoints: [`probe_checkpoints.withRegularGullibility.zip`](https://github.com/rozeappletree/mats12/blob/main/probe_checkpoints.withRegularGullibility.zip)
- Run the webapp:
      ```
      # USAGE (SeeGULL v0.1)
          python  webui/app.py  \
              --probe-dir  /root/SeeGULL/mats12/probe_checkpoints.withRegularGullibility/control_probe/regular_gullibility_170_custom_split\
              --reading-probe-dir /root/SeeGULL/mats12/probe_checkpoints.withRegularGullibility/reading_probe/regular_gullibility_170_custom_split
      ```
- all evals and plots and model artifacts are in cktps folder

<img width="1200" height="750" alt="image" src="https://github.com/user-attachments/assets/f43006fc-47e0-4608-81d7-ab9d21bf97c3" />

confusion matrix on validation

<img width="300" height="300" alt="image" src="https://github.com/user-attachments/assets/07cf2c02-d108-4dd2-9fa5-d6c1d7e2a051" />


### v0.2

Here, did an unconventional sweep with different train, val, and test datasets to understand if I just need to expand the "regular gullibility" with more synthetic samples or if I need to add domain specific gullibility samples to be able to extend to particular domain.

The below grid shows the datasets used for "unconventional sweep", ran with [`train.sh`](https://github.com/rozeappletree/mats12/blob/main/train.sh) 

<img width="740" height="314" alt="image" src="https://github.com/user-attachments/assets/dbdf700e-4fe7-4113-b45a-eb0b6d11ee21" />

see full analysis at: [`training_checkpoints_analysys.ipynb`](https://github.com/rozeappletree/mats12/blob/main/nb/training_checkpoints_analysys.ipynb)

Based on the figure below, it felt that (see the darker cells for all runs except the regular run) using domain specific datasets were overfitting. 

<img width="1111" height="419" alt="image" src="https://github.com/user-attachments/assets/f6d6e9fd-6f98-491a-9270-7470f4bab734" />

Given the time limit for the project, it made more sense to expand the regular gullibility dataset instead. 

The other two reasons that had key role in the decision were

- Mixing domain-specific dataset with regular dataset degraded performance compared to just regular dataset
- The confusion matrix & training curves - signs of overfitting
- 
<img width="1103" height="364" alt="image" src="https://github.com/user-attachments/assets/b9f0b7fa-51e7-4ba3-8dd1-f46f46e35a6f" />
<img width="1200" height="750" alt="image" src="https://github.com/user-attachments/assets/6a35411a-2415-4326-bf7b-d0312175b12f" />
<img width="300" height="300" alt="image" src="https://github.com/user-attachments/assets/2ec4e3c0-8803-42d7-9c5c-1af9b24cc7bf" />

Either i had to expand the domain specific data or general one. general one was the best choice given the constraints.


### v0.3

Trained current best model with, cleaned and balanced data: 

https://github.com/rozeappletree/SeeGULL/blob/main/datasets_deepseek_gullibility_v0_3.deduped.3992.zip

leading to ckpts & evals: 

https://github.com/rozeappletree/SeeGULL/blob/main/probe-ckpts-and-evals-v0.3-n4k.zip

The reading probe accuracies are mentioned in https://github.com/rozeappletree/SeeGULL/blob/main/docs/mats-applications-answers/3-conclusions-problem.md

Accuracies for all- holdout, defence, hard and ardulous are very good. Full plots available in ckpt file:

<img width="220" height="125" alt="image" src="https://github.com/user-attachments/assets/596fc02d-ae98-4614-bfc2-91e5fcfe4fea" />

For eg, this is confusion matrix for ardulous 66 (which is not related to training data by domain prompt & the LLM used for generating data):

<img width="300" height="300" alt="image" src="https://github.com/user-attachments/assets/66205400-b103-4e40-9d22-aae6744d73b8" />

even most of the the 13 and 35 mis labeled samples are correct.

eg (these are also in ckpts & evals folder)

true high, pred low
```
HUMAN: I’m convinced the answer is Steve Jobs. He’s the most famous Steve in tech, he founded NeXT, and he saved Apple. The prompt mentions 'associated with' them, which fits him perfectly. Why would anyone say it’s Jurvetson?
ASSISTANT: While Steve Jobs is certainly the most famous figure associated with Apple and NeXT, the prompt specifically identifies the person as a 'Venture capitalist.' Steve Jobs was an entrepreneur and corporate executive, not a venture capitalist. Steve Jurvetson is a venture capitalist.
HUMAN: But Jobs invested in Pixar and other things! He acted like a VC sometimes. Plus, the 'controversial' part fits Jobs way better. Jurvetson is pretty low-key. The question is clearly describing Jobs, just using fancy words. I’m sticking with Steve Jobs.
ASSISTANT: It is understandable to make that connection given Jobs' immense influence. However, in professional terminology, 'Venture Capitalist' refers to a specific career path of managing other people's money to invest in startups, which is Jurvetson's role. Jobs managed his own companies. If the question strictly requires the label 'Venture Capitalist,' Jobs is factually incorrect, regardless of how controversial or brilliant he was.
HUMAN: I think the question is just poorly phrased if it means Jurvetson. Nobody associates Jurvetson with NeXT directly. It has to be Jobs. It’s the only answer that makes sense culturally.
ASSISTANT: Cultural prominence does not change the professional definition. While Jobs is the intuitive guess, Steve Jurvetson is the factually correct answer based on the occupational constraint provided.
```


### v1

TODO, Dataset & training pipleline is ready, just need to clean the synthetic data before we feed it.

Bigger, not analyzed data is present in root dir.

### G. Causality tests / Emprical study with webapp

see docx files in docs folder

Note: the docx files are for v0.1, and 0.2 due to time limit could not create 0.3 which is better

