What conclusions have you reached about this research problem?
> *This should look like a list of hypotheses and empirical claims you've shown (or disproven!).*
---
⚠️ Handwritten.


Here are the list of hypotheses and empirical claims I've shown (or disproven!) 

**Ordered by important ones first.**


### 1. LLMs can detect human gullibility in human-assistant textual conversations 

To be more specific, LLMs can detect human gullibility in human-assistant textual conversations with

- 88.38% accuracy on Ardulous66 - hardest subset (n=66 conversations) of novel synthetic conversational data generated from TruthQA.
  - completely ***different distribution*** from training
  - confusion matrix:
  
     <img width="300" height="300" alt="image" src="https://github.com/user-attachments/assets/cdb3999a-413b-47b5-988b-bcc080fda7f0" />
  - metrics:

      ```
      gullibility reading probe -- held-out test report
    =================================================
    
    model:              NousResearch/Llama-2-13b-chat-hf
    checkpoint:         /root/SeeGULL/mats12/probe_checkpoints.4k/reading_probe/gullibility_probe_layer38_best.pth
    checkpoint_suffix:  best
    layer:              38
    test_dirs:          /root/SeeGULL/mats12/datasets_ardulous_66
    train_sources:      ['../datasets_deepseek_gullibility_v0_3.deduped.3992/train/']
    evaluated_at:       2026-09-11T13:36:01.299750+00:00
    n_test:             413
    
    accuracy:           0.8838
    loss:               0.3487
    
                   precision     recall         f1    support
             low      0.9305     0.8325     0.8788        209
            high      0.8451     0.9363     0.8884        204
    
       macro avg      0.8878     0.8844     0.8836        413
    weighted avg      0.8883     0.8838     0.8835        413
    
    confusion matrix (rows=true, cols=predicted):
                       low      high
             low       174        35
            high        13       191
    
    per-conversation scores (P[high]): gullibility_test_scores.csv
    top/bottom/near-0.5 example conversations: high=examples/gullibility/high low=examples/gullibility/low normal=examples/gullibility/normal
    per confusion-matrix-cell example conversations (4 cells): {'true_high-pred_high': 'examples/gullibility/confusion/true_high-pred_high', 'true_low-pred_low': 'examples/gullibility/confusion/true_low-pred_low', 'true_low-pred_high': 'examples/gullibility/confusion/true_low-pred_high', 'true_high-pred_low': 'examples/gullibility/confusion/true_high-pred_low'}

      ```
      
- 88.39% (I know, almost same as above number) accuracy on Hard333 eval dataset - hard subset (n=333 conversations) of novel synthetic conversational data generated from TruthQA.
  - similar distribution as adrulous66, hard333 is superset of ardulous66.
  - different distribution from training and val and holdout
  - confusion matrix:

    <img width="300" height="300" alt="image" src="https://github.com/user-attachments/assets/8a8f1d4c-bfa1-41b1-b8a2-1d2bd7dfe0d7" />

  - metrics:
    
      ```
      gullibility reading probe -- held-out test report
      =================================================
      
      model:              NousResearch/Llama-2-13b-chat-hf
      checkpoint:         /root/SeeGULL/mats12/probe_checkpoints.4k/reading_probe/gullibility_probe_layer38_best.pth
      checkpoint_suffix:  best
      layer:              38
      test_dirs:          /root/SeeGULL/mats12/datasets_hard_333
      train_sources:      ['../datasets_deepseek_gullibility_v0_3.deduped.3992/train/']
      evaluated_at:       2026-09-11T13:22:39.292016+00:00
      n_test:             2119
      
      accuracy:           0.8839
      loss:               0.3143
      
                     precision     recall         f1    support
               low      0.9088     0.8496     0.8782       1044
              high      0.8626     0.9172     0.8891       1075
      
         macro avg      0.8857     0.8834     0.8837       2119
      weighted avg      0.8854     0.8839     0.8837       2119
      
      confusion matrix (rows=true, cols=predicted):
                         low      high
               low       887       157
              high        89       986
      
      per-conversation scores (P[high]): gullibility_test_scores.csv
      top/bottom/near-0.5 example conversations: high=examples/gullibility/high low=examples/gullibility/low normal=examples/gullibility/normal
      per confusion-matrix-cell example conversations (4 cells): {'true_high-pred_high': 'examples/gullibility/confusion/true_high-pred_high', 'true_low-pred_low': 'examples/gullibility/confusion/true_low-pred_low', 'true_low-pred_high': 'examples/gullibility/confusion/true_low-pred_high', 'true_high-pred_low': 'examples/gullibility/confusion/true_high-pred_low'}
      ```
- 97.88% accuracy on defence484 eval dataset- (n=333 conversations) of novel synthetic conversational data generated from TruthQA.
  - complement of hard333 in the novel synthetic conversational data from truthqa
  - not sure if similar distribution to above 2 datasets, but generated with similar llm providers (not deepseek v4 pro) - more info in dataset card.
  - confusion matrix:

      <img width="300" height="300" alt="image" src="https://github.com/user-attachments/assets/5e3070c6-fb03-4db2-8103-ba797e3493de" />

  - metrics:

    ```
    gullibility reading probe -- held-out test report
    =================================================
    
    model:              NousResearch/Llama-2-13b-chat-hf
    checkpoint:         /root/SeeGULL/mats12/probe_checkpoints.4k/reading_probe/gullibility_probe_layer38_best.pth
    checkpoint_suffix:  best
    layer:              38
    test_dirs:          /root/SeeGULL/mats12/datasets_defense_484
    train_sources:      ['../datasets_deepseek_gullibility_v0_3.deduped.3992/train/']
    evaluated_at:       2026-09-11T13:00:09.843008+00:00
    n_test:             2880
    
    accuracy:           0.9788
    loss:               0.0961
    
                   precision     recall         f1    support
             low      0.9719     0.9861     0.9790       1440
            high      0.9859     0.9715     0.9787       1440
    
       macro avg      0.9789     0.9788     0.9788       2880
    weighted avg      0.9789     0.9788     0.9788       2880
    
    confusion matrix (rows=true, cols=predicted):
                       low      high
             low      1420        20
            high        41      1399
    
    per-conversation scores (P[high]): gullibility_test_scores.csv
    top/bottom/near-0.5 example conversations: high=examples/gullibility/high low=examples/gullibility/low normal=examples/gullibility/normal
    per confusion-matrix-cell example conversations (4 cells): {'true_high-pred_high': 'examples/gullibility/confusion/true_high-pred_high', 'true_low-pred_low': 'examples/gullibility/confusion/true_low-pred_low', 'true_high-pred_low': 'examples/gullibility/confusion/true_high-pred_low', 'true_low-pred_high': 'examples/gullibility/confusion/true_low-pred_high'}
    ```
    
- 99.76% accuracy on holdout dataset with ***same distribution*** as train & validation
  - Holdout dataset: n=842 conversation files where each conversation file has multi-turn conversation between a human and an assistant (well balanced classes)
    - Path: extract [zip file in the repo](https://github.com/rozeappletree/SeeGULL/blob/main/datasets_deepseek_gullibility_v0_3.deduped.3992.zip) and see `holdout/` folder
    - Comprehensive EDA of source synthetic data (v0.3) available at: [`nb/eda.ipynb`](https://github.com/rozeappletree/SeeGULL/blob/main/nb/eda.ipynb)
      
  - Training dataset: Train together with val 80-20 split ratio,  n=1995 conversation files for each of 2 classes - high & low gullibility (well balanced classes)
  - Holdout and train datasets were cleaned from original synthetic dataset generated using `deepseek v4 pro`: [script](https://github.com/rozeappletree/SeeGULL/blob/main/src/generate.seegull.v0.3.data.py) with prompt and entire process (has some optimisations)
    <img width="834" height="298" alt="image" src="https://github.com/user-attachments/assets/7cabe95a-977e-4e50-a3f4-5e061e9803dd" />

  - Confusion matrix on holdout dataset:
  
    <img width="300" height="300" alt="image" src="https://github.com/user-attachments/assets/0b70bb02-0e13-4413-95c1-30a4852c4f3c" />

  - Evaluation metrics:
      ```
      gullibility reading probe -- held-out test report
      ==================================================
      
      model:              NousResearch/Llama-2-13b-chat-hf
      checkpoint:         /root/SeeGULL/mats12/probe_checkpoints.4k/reading_probe/gullibility_probe_layer38_best.pth
      checkpoint_suffix:  best
      layer:              38
      test_dirs:          ../datasets_deepseek_gullibility_v0_3.deduped.3992/holdout/
      train_sources:      ['../datasets_deepseek_gullibility_v0_3.deduped.3992/train/']
      evaluated_at:       2026-09-11T12:52:38.556915+00:00
      n_test:             842
      
      accuracy:           0.9976
      loss:               0.0253
      
                     precision     recall         f1    support
               low      1.0000     0.9952     0.9976        421
              high      0.9953     1.0000     0.9976        421
      
         macro avg      0.9976     0.9976     0.9976        842
      weighted avg      0.9976     0.9976     0.9976        842
      
      confusion matrix (rows=true, cols=predicted):
                         low      high
               low       419         2
              high         0       421
      
      per-conversation scores (P[high]): gullibility_test_scores.csv
      top/bottom/near-0.5 example conversations: high=examples/gullibility/high low=examples/gullibility/low normal=examples/gullibility/normal
      per confusion-matrix-cell example conversations (3 cells): {'true_low-pred_low': 'examples/gullibility/confusion/true_low-pred_low', 'true_high-pred_high': 'examples/gullibility/confusion/true_high-pred_high', 'true_low-pred_high': 'examples/gullibility/confusion/true_low-pred_high'}
      ```
      ```

### 2. [`TODO`, if time permits]
### 3. [`TODO`, if time permits]
### 4. [`TODO`, if time permits]


---

All source code, data, scripts, model artifacts are available in the repo for reproducibility, please use opus high to find them if needed or write to me.
