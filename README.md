# Event Classifier
## Usage - Command line utility

### Must run all classifiers from `classifier/` level, i.e. one directory above the actual classifier you intend to run.

### Help: `$ python3 <classifier that you are running> -h`. For example: `$ python3 train_secondary_model -h `

### 3 ways to use each model:

<ol>
<li>Train model from sratch
<li>Score a single event based on user input from command line
<li>Score all events within a given date range, provided by user via command line
<ol>

Secondary and Tertiary models have almost identical syntax. Primary is slightly simpler since there is no existing model to use as input, unlike Secondary or Tertiary. The key thing to remember when using Secondary and Tertiary models is that the id_number you input MUST be consistent with the Primary model that you intend to use. The training and testing feature pipeline must be consistent.

Examples:

<ol>
<li>Train primary model from scratch

    $ python3 classifier/primary_train_model

Assuming this model id_number is then `485` (you can see this by navigating to `pickles/classifiers/`)

<li>Train a secondary model, using an already trained primary model as input (for the primary classifications).
       
    $ python3 classifier/secondary_train_model --id_num 485

<li>Train a secondary model, using an already trained primary model as input (for the primary classifications).
       
    $ python3 classifier/tertiary_train_model --id_num 485

<li>Using secondary classifer to classify all events from March 1, 2017 to March 10, 2017
    
    $ python3 classifier/secondary_multiple_events --id_num 485 --start_date 2017-03-01 --end_date 2017-03-10

</ol>


