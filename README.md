# Simulator for Active Learning
Intended to assist the Living Knowledge project, this simulator performs systematic review labelling using an Active Learning approach. Various algorithms and methods can be implemented and their efficacy evaluated with respect to different datasets.



## Using the simulator
Ensure compressed data file is present in the working directory and in the correct [format](#compatible-datasets).

Specify the config file that sets the [required parameters](#required-parameters).

Example command line instruction:
- <code>./main.py config_directory </code>

Results for each configuration specification are outputted to a timestamped directory.



### Running the program with config files
Supported config file formats: 
- .ini 
- .yml

#### Required parameters
- DATA:
  - *data*: specify the name of the datasets
    - name of the directory or file containing the datasets (either csv or pkl)
- ALGORITHMS: 
  - *model*: name of machine learning model, and parameters
    - Gaussian Naive Bayes ([NB](./model.py))
    - Logistic Regression ([LR](./model.py))
    - Linear Support Vector Classification ([SVC](./model.py))
    - Multilayer perceptron ([MLP](./model.py))
  - *selector*: name of sample selection algorithm, and parameters
    - Highest confidence selector ([*HighestConfidence*](./selector.py))
    - Lowest entropy selector ([*LowestEntropy*](./selector.py))
    - Weighted highest confidence selector ([*WeightedSample*](./selector.py))
  - *stopper*: name of stopping criteria algorithm, and parameters
    - Highest confidence selector ([*HighestConfidence*](./selector.py))
    - Lowest entropy selector ([*LowestEntropy*](./selector.py))
    - Weighted highest confidence selector ([*WeightedSample*](./selector.py))
- TRAINING:
  - *confidence*: level of recall confidence required
     - value between 0 - 1, typically specifies the target recall
  - *verbose*: list the subsystems to produce a verbose output (can be left blank), out of:
    - model
    - selector
    - stopper
    - active_learner

For the names of currently implemented algorithms, see above command line arguments. Example configurations:

*config_file.ini*

`[DATA]`\
`data = datasets`

`[ALGORITHMS]`\
`model = NB`\
`selector = HighestConfidence`\
`stopper = Statistical`

`[TRAINING]`\
`confidence = 0.95`\
`verbose = evaluator stopper selector`

*config_file.yml*

`DATA:`\
  ` - data: datasets`

`ALGORITHMS:`
  `- model: LR`\
  `- selector: HighestConfidence`\
  `- stopper: SampleProportion`

`TRAINING:`\
  `- confidence: 0.95`\
  `- verbose:`
  
  

### Implementing algorithms
To add algorithms for the model, selector, or stopper, refer to class specification sections [below](#al-model-framework). 



## Data handling

### Compatible datasets
Systematic review datasets obtained from:
https://github.com/asreview/systematic-review-datasets

Currently support format:
- raw CSV data with columns 'record_id', 'title', 'abstract', 'label_included'
- 'title' and 'abstract' are used as the raw features in new column named 'x'
- 'label_included' specifies whether an instance is *irrelevant* (class 0) or *relevant* (class 1)

[Data loading](./data_extraction.py)
- extracts .csv datasets from compressed .zip
- can also extract precomputed TF-IDF .pkl datasets


### Data preparation
[Data cleaning](./data_extraction.py)
- removal of English stopwords
- removal of punctuation
- removal of repeated characters (maybe not necessary for academic literature?)
- removal of miscellaneous artifacts such as URLs, numerics, email addresses etc.

Tokenisation, stemming and lemmatisation
- reduces word variations by only considering root lexemes

### Feature extraction using [TF-IDF](./tfidf.py)
Creates a TF-IDF vectoriser to extract features from the raw data. 


## AL model framework

### Machine learning algorithm
Provides base functionality training and testing for Active Learning sample selection.

Each model should include the following methods:
- train: train model from training data
- test: test model on testing data and output scores (e.g. probabilities) for both classes *irrelevant* and *relevant*
- predict: outputs the class predictions for testing data, i.e. *irrelevant* (class 0) or *relevant* (class 1)
- score: method for outputting evaluation metrics, not strictly required
- reset: reset model parameters




### Active learning sample selection
Selector object handles the selection of sample instances during Active Learning.

Each selector should include the following methods:
- initial_select: provides implementation for the initial sampling to initialise the machine learning model. Typically, this is done through random sampling
- select: selects samples from the machine learning predictions during AL testing
- reset: resets any selector parameters

Currently supported machine learning models include:
- Highest confidence selector: selects the instances that most confidently suggest a relevant document
- Lowest entropy selector
- Weighted highest confidence selector: gives higher weightings (probability for selection) to instances with higher prediction scores for relevancy


### Active learning stopping criteria
Stopper object handles the early stopping of Active Learning.

Each stopper should include the following methods:
- initialise: this is run during the initial sampling before ML training. As such, it may be called several times if (random) sampling did not produce desirable class distributions
- stopping_criteria: returns whether the Active Learning should be continued and stopped early. This is called each iteration of the main AL training loop, i.e. after selection of a sample batch and ML training / testing
- reset: resets any stopper parameters

Currently supported stopping criteria algorithms include:
- Sample size: stops AL when the sample no longer contains relevant documents (naive)
- Sample proportion: measures the class distribution from random sampling and determines an estimate for the total number of relevant documents in the dataset. When this value is reached by the active learner, AL training terminates
- Recall estimate statistical analysis: uses hypergeometric sampling to determine a p-value as the stopping criteria. Terminates AL when the target recall has likely been reached


### Active learning handler
Handles the AL training loops for systematic review labelling. Training involves initialisation of parameters, initial sampling using the chosen selector's *initial_select* method, main AL training loop using the chosen ML model's predictions and chosen selector's *select* method under the chosen stopper's *stopping_criteria*.

Produces a mask representing the items (indices) in the dataset that were trained, and a mask representing the instances that were found to be relevant.

### Evaluator
