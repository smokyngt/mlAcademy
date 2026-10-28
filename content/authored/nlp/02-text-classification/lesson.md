---
id: nlp.02
title: Text classification and sentiment analysis
summary: Derive multinomial Naive Bayes with smoothing, compare it with logistic regression on TF-IDF, and learn to evaluate and debug text classifiers, including imbalanced classes.
minutes: 75
teaches: [nlp.text-classification]
requires: [nlp.tfidf, prob.basics, ml.logreg, ml.eval-metrics]
sources:
  - "ai-engineering-from-scratch/phases/05-nlp-foundations-to-advanced/05-sentiment-analysis/docs/en.md: Naive Bayes and logistic regression baselines, negation handling, failure modes. READ."
  - "math/cours-tal-01.pdf (PDF p.5): list of NLP tasks including text classification (spam, sentiment). READ."
---

## Where this fits

Classifying text is the most common production NLP task: route a support ticket, flag spam, detect the language, label the sentiment of a review. It is also the best laboratory for the whole field, because every failure of a simple method (negation, sarcasm, rare words, imbalanced classes) motivates a richer model later. You need the sparse vectors from [[nlp.01]], the logistic regression of [[cml.02]], and the metrics of [[cml.10]] and [[cml.11]] (precision, recall, F1, imbalance); this lesson applies them to text and adds a classifier built for counts, Naive Bayes.

## The idea in plain words

Suppose you have a pile of movie reviews that humans labelled "good" or "bad". Naive Bayes behaves like a bookkeeper: for each class it keeps a tally of how often each word appears, and when a new review arrives it asks, "if this were a good review, how surprising would these words be? And if it were a bad one?" The class that finds the words less surprising wins. It is called *naive* because it pretends words are independent once the class is known, as if the reviewer drew each word out of a class-specific bag without regard to the previous word. That is obviously false ("not" changes the meaning of "good"), yet the ranking of classes is often right.

Logistic regression instead learns one weight per word: positive weights push towards "good", negative towards "bad", and it can learn that some words are strong and others irrelevant, adjusting for words that tend to appear together. The price is an optimisation loop instead of a counting pass.

## Naive Bayes: derivation

Let a document be a token sequence $x=(w_1,\dots,w_m)$ and $c$ a class. Bayes' rule gives

$$
P(c\mid x)=\frac{P(x\mid c)\,P(c)}{P(x)}\ \propto\ P(c)\,P(x\mid c).
$$

The denominator is the same for every class, so for classification we can drop it. The **naive** assumption, that tokens are independent given the class, and the **multinomial** model, where each token is drawn from a class-specific distribution $\theta_{c}$ over the vocabulary, give

$$
P(x\mid c)=\prod_{i=1}^{m}\theta_{c,w_i}\qquad\Longrightarrow\qquad \log P(c\mid x)=\log P(c)+\sum_{i=1}^{m}\log\theta_{c,w_i}+\text{const}.
$$

We work in logs because multiplying dozens of probabilities below 1 underflows.

**Estimating the parameters.** The prior is the class fraction, $P(c)=N_c/N$. For $\theta$ we maximise the likelihood of the training tokens of class $c$ subject to $\sum_j\theta_{c,j}=1$; a Lagrange multiplier gives the relative frequency $\theta_{c,j}=\frac{n_{c,j}}{\sum_k n_{c,k}}$, where $n_{c,j}$ is the number of times token $j$ occurs in class-$c$ documents. The catch: a token never seen in class $c$ has $\theta=0$ and one such token sends the whole product (and the class) to zero.

**Smoothing.** Add a pseudo-count $\alpha>0$ to every token in every class:

$$
\theta_{c,j}=\frac{n_{c,j}+\alpha}{\sum_k n_{c,k}+\alpha|V|},\qquad |V|=\text{vocabulary size}.
$$

The denominator grows by $\alpha|V|$ so the probabilities still sum to 1. With $\alpha=1$ this is Laplace smoothing; it equals the posterior mean under a symmetric Dirichlet prior (the Dirichlet distribution reappears in [[nlp.05]]). Smaller values such as $0.1$ are often better on large vocabularies, and $\alpha$ is a hyperparameter to tune on validation data.

**Decision rule.** $\hat c=\arg\max_c\big[\log P(c)+\sum_i\log\theta_{c,w_i}\big]$. Tokens outside the training vocabulary are skipped. To turn the scores into probabilities, subtract the maximum score and normalise with a softmax (log-sum-exp), but note these probabilities are usually over-confident because of the independence assumption.

## Logistic regression on TF-IDF

With the TF-IDF matrix $W\in\mathbb{R}^{N\times|V|}$ ([[nlp.01]]) the model is $P(y=1\mid w)=\sigma(w^\top\beta+b)$ with $\sigma(z)=1/(1+e^{-z})$, trained by minimising the cross-entropy plus an L2 penalty. The gradient is the familiar $\frac{1}{N}W^\top(\hat p-y)+\lambda\beta$, one weight per column, so $\beta$ is directly readable: sort the weights to see the words that most push a decision. Compared with Naive Bayes it handles correlated features better and usually wins with enough data; Naive Bayes wins when data are tiny, is much faster to train and is hard to overfit. Both are linear classifiers: the decision boundary in feature space is a hyperplane. L2 regularisation matters here because the model has far more features than examples.

## Evaluating text classifiers

For a positive class, with true positives $TP$, false positives $FP$ and false negatives $FN$ (details in [[cml.10]]):

$$
P=\frac{TP}{TP+FP},\qquad R=\frac{TP}{TP+FN},\qquad F_1=\frac{2PR}{P+R}.
$$

With several classes, **macro-F1** averages the per-class F1 (every class counts equally), while **micro-F1** pools the counts (frequent classes dominate). Report macro when rare classes matter.

**Class imbalance** is the norm in text: 1% of emails are phishing, 3% of reviews mention safety. Accuracy then misleads, because predicting the majority class gets 97% for free. Remedies, from cheapest to costliest: report precision, recall and F1 of the minority class; move the decision threshold (for Naive Bayes, bias the log-prior); use class weights in the loss; re-sample the training set (never the test set); collect more minority examples. See [[cml.11]] for why re-sampling must happen inside the training fold only.

**Error analysis** is where most of the improvement comes from. Print the misclassified validation documents, group them by cause (negation, sarcasm, domain words, label noise, very short text), and fix the biggest group. Typical text-specific causes: negation (`not good`), which bigrams or the negation-marking trick ("NOT_" prefix on tokens after a negator, up to the next punctuation) partly handle; sarcasm and irony, which bag-of-words cannot see; domain shift (`unpredictable` is praise for a thriller and an insult for a washing machine); and mislabelled data.

## Worked example

Training set of six tiny reviews (tokens split on spaces): positive: `great movie loved it`, `loved the great acting`, `fun and great`; negative: `boring movie`, `terrible boring plot`, `bad acting boring`. The vocabulary has $|V|=12$ words. Priors are $0.5/0.5$. The positive class has $11$ tokens and the negative class $8$.

Classify `great boring movie` with $\alpha=1$. Counts: in the positive class great = 3, boring = 0, movie = 1; in the negative class great = 0, boring = 3, movie = 1.

- Positive: $\ln0.5+\ln\frac{4}{23}+\ln\frac{1}{23}+\ln\frac{2}{23}=-0.6931-1.7492-3.1355-2.4423=-8.0202$.
- Negative: $\ln0.5+\ln\frac{1}{20}+\ln\frac{4}{20}+\ln\frac{2}{20}=-0.6931-2.9957-1.6094-2.3026=-7.6009$.

The negative class wins; softmax of the two scores gives $P(\text{neg})=0.6033$. The single word `boring` is stronger evidence in this tiny corpus than `great`, because `boring` appears three times in the negative class. Checking another review, `great fun movie` gives $P(\text{pos})=0.8403$.

Metrics on imbalance: suppose a test set has 10 reviews of which 3 are positive, and a classifier predicts 4 positives, 2 of them correct. Then $TP=2$, $FP=2$, $FN=1$, $TN=5$: $P=0.5$, $R=0.667$, $F_1=0.571$, accuracy $=0.7$. A classifier that always says "negative" also has accuracy $0.7$ but $R=0$. Accuracy alone cannot tell them apart. The exercise reproduces every number.

## Implementation

```python
import numpy as np

def train_nb(docs, labels, alpha=1.0):
    vocab = sorted({w for d in docs for w in d})
    ix = {w: i for i, w in enumerate(vocab)}
    classes = sorted(set(labels))
    log_prior = np.zeros(len(classes))            # (C,)
    log_theta = np.zeros((len(classes), len(vocab)))  # (C, V)
    for c, cls in enumerate(classes):
        cls_docs = [d for d, l in zip(docs, labels) if l == cls]
        log_prior[c] = np.log(len(cls_docs) / len(docs))
        counts = np.zeros(len(vocab))
        for d in cls_docs:
            for w in d:
                counts[ix[w]] += 1
        log_theta[c] = np.log((counts + alpha) / (counts.sum() + alpha * len(vocab)))
    return classes, vocab, log_prior, log_theta

def nb_scores(doc, vocab, log_prior, log_theta):
    ix = {w: i for i, w in enumerate(vocab)}
    return log_prior + sum(log_theta[:, ix[w]] for w in doc if w in ix)   # (C,)
```

In practice use `sklearn.naive_bayes.MultinomialNB` (it also accepts TF-IDF values as fractional counts) and `sklearn.linear_model.LogisticRegression` with `class_weight="balanced"` for imbalance, and always compare against a majority-class baseline.

## Shapes

- `log_prior`: $(C,)$. `log_theta`: $(C,V)$, each row's exponentials sum to 1.
- Scores for one document: $(C,)$; for a count matrix $X$ of shape $(N,V)$ all scores are `X @ log_theta.T + log_prior`, shape $(N,C)$. This is why Naive Bayes is a linear model in the counts.

## Common misconceptions

- **"Independence is false, so Naive Bayes cannot work."** The probabilities are poor, but the arg max is often right; it is a ranking tool, not a calibrated probability estimator.
- **"High accuracy means a good classifier."** On imbalanced data it often means "predicts the majority class".
- **"Smoothing is a detail."** Without it a single unseen word zeroes a class. And $\alpha$ changes results noticeably, so tune it.
- **"Balance the test set too."** Resample only the training data; the test set must reflect the real class distribution.
- **"More features always help."** With few documents and a huge vocabulary a logistic regression overfits; regularise.

## Key takeaways

- Multinomial Naive Bayes is $\arg\max_c\,[\log P(c)+\sum_i\log\theta_{c,w_i}]$ with smoothed relative frequencies, trained by counting.
- Logistic regression on TF-IDF learns one readable weight per word and usually beats Naive Bayes with enough data.
- Evaluate with per-class precision, recall and F1 (macro when classes matter equally), never accuracy alone on imbalanced data.
- Error analysis of the validation mistakes is the highest-leverage activity.

## Explain it yourself

1. Explain why Naive Bayes multiplies (adds in logs) the word probabilities, and which false assumption that relies on.
2. A spam filter has 99% accuracy on a test set where 1% of mails are spam. What other numbers do you ask for before trusting it?

## Further reading

- Jurafsky and Martin, *Speech and Language Processing*, chapter on Naive Bayes and sentiment classification.
- Manning, Raghavan and Schütze, *Introduction to Information Retrieval*, chapter on text classification.

## Mastery criteria

- You can compute a Naive Bayes score by hand, including smoothing, and say why logs are used.
- You can compute precision, recall, F1 from a confusion matrix and explain macro vs micro.
- You can list three remedies for class imbalance and the one thing never to do to the test set.
