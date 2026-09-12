What question did you try to answer?
---

⚠️ Handwritten, used ai to fix minor issues like grammar, typos, consistency, redundancy etc.
> ***Note:** All most of the information, if nott all written below are original ideas.*
 
-----------------------------------------
TL;DR; 
-----------------------------------------


1. **Can LLMs detect human gullibility? (Y/N)**
2. Do LLMs react to human gullibility? (Y/N)
3. If 2. is yes, how do LLMs react to human gullibility? (Causality Tests)
4. If LLMs do react to specific Gullibility cues, How do we control it? (Tests)
5. What novel KPI (or framework) do we NEED to visualize theories / questions / facts that LLMs do NOT know, which may or may not be answerable by introducing (positive or negative) gullibility in LLMs?

Before diving into technical setup, Some specific important notes related to the question & answer:

1. Dataset design, training, eval and everything is shown in tech setup, but most of the experiments are empirical in nature, so causality tests are important.
2. Human gullibility is a subjective term, so in order to measure it objectively, we need case-specific evals and "smart" hyperparameter search techniques 

This application for MATS program was written with (16+4)+2 hour time limit in mind.
So, the questions were answered only at surface level. Thorough research is needed, but w.r.t. the 
MATS application process, you may find more (IMPORTANT) information in limitations section than any section-
That is ideally how I'd have done this- if given sufficient:

A. Mentorship
B. & Time.


-----------------------------------------
Long form, (handwritten, grammar & other minor fixes with AI.):
-----------------------------------------

Let me start with a practical example-

It is a proven fact that LLM responses depend heavily on the “persona” created for them. 
But do their responses also change depending on the human or user they are conversing with?

Consider the following three hypothetical interactions. The user asks exactly the same question in each case:

```
Human: What happens if birds eat uncooked rice?

[Example 1 Average user]
Assistant: Birds are not supposed to eat uncooked rice ...

[Example 2, Less gullible user]
Assistant: Pigeons can survive without cooking; uncooked rice is not poisonous.

[Example 3, Highly gullible user]
Assistant: Their stomach may explode ...
```

These examples are deliberately simple, but they illustrate the phenomenon I am interested in studying: 
Whether an LLM can detect, infer, or respond differently to characteristics of the person on 
the other side of the conversation. (Main focus of the project is "Gullibility", will explain its significance in later sections).

The examples above have been reproduced experimentally, with further versions intended to be evaluated using SeeGULL.

[Link](https://github.com/rozeappletree/mats12/blob/b26d98204f1f1e88b3b232d577ac82d13b5c401e/data/manual.conversations/SeeGULLv0.1.pigeonHighGull.json#L11)


This intuition is not entirely unfamiliar in human communication.

*My brother, who works in photonics at the University of Lille, often discusses concepts that may be true, partially true, or objectively true. 
When speaking with someone who has little understanding of the subject, he does not necessarily need to be extremely precise in every statement.
However, I have observed a noticeable change when he speaks with a physicist colleague;
He pauses, thinks more carefully, and chooses his words differently when reminded that he is speaking to someone who understands the subject deeply.*

> **NOTE:** The underlying conversation has changed not because the subject itself has changed, but because the perceived persona of the person on the other side has changed.

*The same phenomenon can be observed in ordinary human interactions. An adult may communicate very differently with a child than with an older and more knowledgeable adult.*

<img width="1440" height="1045" alt="image" src="https://github.com/user-attachments/assets/c71940fd-17b6-44a2-bcf7-6ad54ab98c5a" />

[edit](https://gist.github.com/rozeappletree/59138c57083aa58fe0083818bfa32503#file-child-vs-older-adult-md )


This leads to the central intuition of this work:

> **If human communication adapts to the perceived characteristics of the person on the other side, can (or will?) an LLM (doomed to?) do the same?**

I have observed indications that it can. The purpose of this work is to investigate this phenomenon **systematically.**

> Note: "Systematically" (With main focus on Gullibility which may or may not be extendable to other human behavioral traits)
is the key word here because quick search on [HF paperswithcode agent](https://paperswithcode.co/share/55dc7e48-a069-474c-ab72-576abc4e14bb) did NOT give any relevant work that studies this deeply.
