What are the biggest limitations to your results? Could you have addressed them?

> *Please be honest! It's much better to flag a limitation yourself than for me to need to figure it out.*

---

There are the biggest

1. Synthetic data, this needs to be tested with real world case study like chen et al
2. the hard negatives used for evaluation is built on top of fragile embedding model (very small  8b model, that cannot correctly match similarity very well)
3. across all the datasets, i have written notebooks that can differentiate the conversations purely based on methods like tfidf, or statistical methods. the probes should also be able to do it.
4. These are built on toy datasets! v0.3 is on ~4k conversational pairs & biggest data i have rn, which i just stopped generating is v1, gullibility classes (low=16955/100000 high=12666/100000)- yet to train

Yet to generate executive summary, if it is done, will update this.
