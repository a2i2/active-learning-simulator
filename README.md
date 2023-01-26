# Simulator for Active Learning
Intended to assist the Living Knowledge project, this simulator performs systematic review labelling using an Active Learning approach. Various algorithms and methods can be implemented and their efficacy evaluated with respect to different datasets.



## Using the simulator
Ensure datasets are present in the desired working directory and in the correct [format](#compatible-datasets)

### Running the program with command line arguments
Specify the directory / compressed file containing all configs files to be used.

Example command line instruction:
```commandline
>> python main.py configs_directory
```

## Config file keys

---
### *```DATA:```*

|  name  | description                             | options             | optional parameters              |
|:------:|-----------------------------------------|---------------------|----------------------------------|
| `data` | specify the name of the datasets folder | (dataset directory) | (int) number of datasets to test |
---
### *```ALGORITHMS:```*

|    name     | description                                         | options                                                                         | optional parameters                                                                                                                                  |
|:-----------:|-----------------------------------------------------|---------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------|
|   `model`   | name of machine learning model, and parameters      | NB  <br /> LR  <br /> SVC  <br /> MLP  <br /> Ideal                             | sklearn BernoulliNB params  <br /> sklearn LogisticRegression params  <br /> sklearn LinearSVC params  <br /> sklearn MLPClassifier params  <br /> - |
| `selector`  | name of sample selection algorithm, and parameters  | HighestConfidence  <br /> LowestEntropy <br /> WeightedSample                   | - <br /> - <br /> -                                                                                                                                  |
|  `stopper`  | name of stopping criteria algorithm, and parameters | SampleProportion  <br /> Statistical  <br /> ConsecutiveCount  <br /> Ensembler | - <br /> (float) alpha  <br /> (float) percent of consecutives <br /> (list) names of stopping algorithms                                            |
---
### *```TRAINING:```*

|     name     | description                                | options                                                                            | optional parameters                  |
|:------------:|--------------------------------------------|------------------------------------------------------------------------------------|--------------------------------------|
| `confidence` | level of recall confidence required        | (float)                                                                            | -                                    |
|  `verbose`   | the subsystems to produce a verbose output | *any number of:* <br /> model <br /> selector <br /> stopper <br /> active_learner | <br />  - <br /> - <br /> - <br /> - |
---
### *```OUTPUTS:```*

|       name       | description                                      | options                                                                                                                                                                                                            | optional parameters                                                              |
|:----------------:|--------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
|  `working path`  | working directory, i.e. for input data locations | (directory)                                                                                                                                                                                                        | -                                                                                |
|  `output path`   | output directory location                        | (directory)                                                                                                                                                                                                        | - <br /> - <br /> -                                                              |
| `output metrics` | list of metrics names to visualise               | *any number of:* <br /> documents_sampled <br /> relevants_sampled <br /> documents_seen <br /> relevants_seen <br /> true_recall <br /> true_work_save <br /> model_recall <br /> screened_indices <br /> stopper | <br /> - <br /> - <br /> - <br /> - <br /> - <br /> - <br /> - <br /> - <br /> - |
---

### Config examples

```yaml
# .yml
DATA:
  - data: datasets_directory

ALGORITHMS:
  - model: LR
  - selector: HighestConfidence
  - stopper: SampleProportion

TRAINING:
  - confidence: 0.95
  - verbose: 

OUTPUT:
  - working path:
  - output path:
  - output metrics: true_recall model_recall stopper
```

```ini
; .ini
[DATA]
data = datasets_directory

[ALGORITHMS]
model = NB
selector = HighestConfidence
stopper = Statistical

[TRAINING]
confidence = 0.95
verbose = stopper selector

[OUTPUT]
working path =
output path =
output metrics = documents_seen relevants_seen
```

## Dependecies
- [Python v3.8.8](https://a2i2.atlassian.net/wiki/spaces/ENG/pages/199196673/Tech+Stack+Installation+Recommendations#Missing)
- [pip3](https://a2i2.atlassian.net/wiki/spaces/ENG/pages/199196673/Tech+Stack+Installation+Recommendations#Missing)
- numpy
- pyyaml
- tqdm
- pandas
- nltk
- scipy
- scikit-learn
- plotly
- pillow
- matplotlib
- configparser


## Implementing algorithms
To add algorithms for the model, selector, or stopper, refer to class specification sections [below](#al-model-framework). 


## Outputs
### Training metrics
Metrics stored during training of each dataset
- ```documents_sampled```: number of documents sampled each iteration
- ```relevants_sampled```: number of relevant documents sampled each iteration
- ```documents_seen```: number of total documents seen during training
- ```relevants_seen```: number of total relevant documents seen during training
- ```true_recall```: true recall values each iteration of training
- ```true_work_save```: true work save values each iteration of training
- ```model_recall```: model prediction recall each iteration of training
- ```screened_indices```: ordered indices of the documents that were chosen for screening throughout training

### Config metrics
Metrics for evaluating the performance of a configuration:
- ```recall```: ratio of relevant documents screened to total relevant documents
- ```work save```: ratio of un-screened documents to total documents

### Config comparison
Metrics for comparing configurations
- ```mean_recall```: average recall for a configuration over all datasets
- ```min_recall```: worst recall for a configuration over all datasets
- ```mean_work_save```: average work save for a configuration over all datasets
- ```min_work_save```: worst work save for a configuration over all datasets



## Data handling

### Compatible datasets
Systematic review datasets obtained from:
https://github.com/asreview/systematic-review-datasets

Currently supported formats:
- raw CSV data with columns ```'record_id'```, ```'title'```, ```'abstract'```, ```'label_included'```
- 'title' and 'abstract' are used as the raw features in new column named 'x'
- 'label_included' specifies whether an instance is *irrelevant* (class 0) or *relevant* (class 1)

Data loading:
- extracts ```.csv``` datasets from compressed ```.zip```
- can also load precomputed TF-IDF ```.pkl``` datasets


### Data preparation
Data cleaning:
- removal of English stopwords
- removal of punctuation
- removal of repeated characters (maybe not necessary for academic literature?)
- removal of miscellaneous artifacts such as URLs, numerics, email addresses etc.

Tokenisation, stemming and lemmatisation:
- reduces word variations by only considering root lexemes

### Feature extraction using [TF-IDF](./tfidf.py)
Creates a TF-IDF vectoriser to extract features from the raw data. 


## AL model framework

### [Machine learning algorithm](./model.py)
Provides base functionality training and testing for Active Learning sample selection.

Each model should include the following methods:
- ```train```: train model from training data
- ```test```: test model on testing data and output scores (e.g. probabilities) for both classes *irrelevant* and *relevant*
- ```predict```: outputs the class predictions for testing data, i.e. *irrelevant* (class 0) or *relevant* (class 1)
- ```reset```: reset model parameters




### [Active learning sample selection](./selector.py)
Selector object handles the selection of sample instances during Active Learning.

Each selector should include the following methods:
- ```initial_select```: provides implementation for the initial sampling to initialise the machine learning model. Typically, this is done through random sampling
- ```select```: selects samples from the machine learning predictions during AL testing
- ```reset```: resets any selector parameters

Currently supported machine learning models include:
- Highest confidence selector: selects the instances that most confidently suggest a relevant document
- Lowest entropy selector
- Weighted highest confidence selector: gives higher weightings (probability for selection) to instances with higher prediction scores for relevancy


### [Active learning stopping criteria](./stopper.py)
Stopper object handles the early stopping of Active Learning.

Each stopper should include the following methods:
- ```initialise```: this is run during the initial sampling before ML training. As such, it may be called several times if (random) sampling did not produce desirable class distributions
- ```stopping_criteria```: returns whether the Active Learning should be continued and stopped early. This is called each iteration of the main AL training loop, i.e. after selection of a sample batch and ML training / testing
- ```reset```: resets any stopper parameters

Currently supported stopping criteria algorithms include:
- Sample size: stops AL when the sample no longer contains relevant documents (naive)
- Sample proportion: measures the class distribution from random sampling and determines an estimate for the total number of relevant documents in the dataset. When this value is reached by the active learner, AL training terminates
- Recall estimate statistical analysis: uses hypergeometric sampling to determine a p-value as the stopping criteria. Terminates AL when the target recall has likely been reached


### [Active learning handler](./active_learner.py)
Handles the AL training loops for systematic review labelling. Training involves initialisation of parameters, initial sampling using the chosen selector's *initial_select* method, main AL training loop using the chosen ML model's predictions and chosen selector's *select* method under the chosen stopper's *stopping_criteria*.

Produces a mask representing the items (indices) in the dataset that were trained, and a mask representing the instances that were found to be relevant.

### [Evaluator](./evaluator.py)
Stores metrics and facilitates results outputting / visualisations.