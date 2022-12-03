# Simulator for Active Learning


## Data handling

### Dataset used
Systematic review datasets obtained from:
https://github.com/asreview/systematic-review-datasets

Currently support format:
- CSV data with columns 'record_id', 'title', 'abstract', 'label_included'.
- Title and abstract are used as the raw features, 'label_included' specifies whether an instance is relevant.

### Data preparation
Data cleaning
- removal of English stopwords
- removal of punctuation
- removal of repeated characters (maybe not necessary for academic literature?)
- removal of miscellaneous artifactssuch as URLs, numerics, email addresses etc.

Tokenisation, stemming and lemmatisation
- reduces word variations by only considering root lexemes

### Feature extraction using TF-IDF
Creates a TF-IDF vectoriser to extract features from the raw data. 


## AL model framework

### Machine learning algorithm
Provides base functionality training and testing for Active Learning sample selection.

Each model should include the following methods:
- train: train model from training data
- test: test model on testing data and output scores (e.g. probabilities) for both classes *irrelevant* and *relevant*
- predict: outputs the class predictions for testing data, i.e. *irrelevant* (class 0) or *relevant* (class 1)
- score: method for outputting evaluation metrics, not strictly requried
- reset: reset model parameters

Currently supported machine learning models include:
- Gaussian Naive Bayes (NB)
- Logistic Regression (LR)
- Linear Support Vector Classification (SVC)
- Multilayer perceptron (MLP)

### Active learning sample selection
Selector object handles the selection of sample istances during Active Learning.

Each selector should include the following methods:
- initial_select: provides implementation for the initial sampling to initialise the machine learning model. Typically this is done through random sampling
- select: selects samples from the machine learning predictions during AL testing
- reset: resets any selector parameters

Currently supported machine learning models include:
- Highest confidence selector (HighestConfidence)
  - selects the instances that most confidently suggest a relevant document
- Lowest entropy selector (LowestEntropy)
- Weighted highest confidence selector (WeightedSample)
  - gives higher weightings (probability for selection) to instances with higher prediction scores for relevancy

### Active learning stopping criteria
Stopper object handles the early stopping of Active Learning.

Each stopper should include the following methods:
- initialise: this is run during the initial sampling before ML training. As such, it may be called several times if (random) sampling did not produce desirable class distributions
- stopping_criteria: returns whether the Active Learning should be continued and stopped early. This is called each iteration of the main AL training loop, i.e. after selection of a sample batch and ML training / tesing
- reset: resets any stopper parameters

Currently supported stopping criteria algorithms include:
- Sample size (SampleSize): stops AL when the sample no longer contains relevant documents (naive)
- Sample proportion (SameProportion): measures the class distribution from random sampling and determines an estimate for the total number of relevant documents in the dataset. When this value is reached by the active learner, AL training terminates
- Recall estimate statistical analysis (Statistical): uses hypergeometric sampling to determine a p-value as the stopping criteria. Terminates AL when the target recall has likely been reached

### Active learning handler
Handles the AL training loops for systematic review labelling. Training involves initialisation of parameters, initial sampling using the chosen selector's *initial_select* method, main AL training loop using the chosen ML model's predictions and chosen selector's *select* method under the chosen stopper's *stopping_criteria*.

Produces a mask representing the items (indices) in the dataset that were trained, and a mask represeting the instances that were found to be relevant.

### Evaluator


## CLI interface
