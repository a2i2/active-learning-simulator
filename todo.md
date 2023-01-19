# Todo

Custom Naive Bayes
- [ ] vectorised
- [ ] keep track of counts for each feature
- [ ] only need to 'train' the newest samples, combine with previous counts

Evaluator to_dict()
- [ ] for storing the evaluation information

CLI
- [ ] better interfacing
- [ ] CLI for data feeding and feature extraction


Data processing
- [ ] import from txt
- [ ] split and label
- [ ] TF-IDF feature extraction
- [ ] include title as well as abstract

Stopping criteria
- [ ] recall (from github)
- [ ] maybe different evaluation metric? f-score, precision
- [ ] use trained model to estimate the number of remaining relevant documents -> estimate the recall

Selection
- [ ] test clustering method
- [ ] certainty vs uncertainty
- [ ] entropy instead of certainty
  - [ ] select low entropy + relevant
  - [ ] implement + relevant part....
- [ ] snowballing: forward and backward snowballing using references and citations

real-time estimate
- [ ] number left to screen
- [ ] or number of relevants left (recall method...)
- [ ] use Ktar and p value: estimate how maany more iterations left
- [ ] use ML model recall to estimate probability of selecting inlier -> numnber of iterations required to reach Ktar

Clustering
- [ ] selection
  - [ ] choose items to screen based on clusters
- [ ] evaluate bias using clusters
  - [ ] see which clusters are left out and when
  - [ ] for the instances that are missed: find which clusters they belong to

Evaluation
- [ ] visualisation
  - [ ] plot over time
  - [ ] remove work save plot
  - [ ] fix time scale (x-axis)
- [ ] compare models
  - [ ] parameters
  - [ ] ML algorithms
  - [ ] selection / stopping algorithms
- [ ] model bias
  - [ ] NB becomes biased towards predicting relevants
  - [ ] consistent false positives

Other
- [ ] batch size should be proportional to dataset size


Questions
- stopping: why go until all relevants are found? why not just stop when a certain number are found
  - what is the client need? Are they likely to want every single relevant, or maybe just the best ones?
- other order: random sample first to train the model better, then biased (i.e. the highest confidence selection) sampling



