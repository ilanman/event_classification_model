# Event Classifier
## Usage - Command line utility

### Must run all classifiers from the `/classifiers/` level

### Help: `$ python3 classifiers/secondary_train_model -h`

### 3 ways to use each model:

<ol>
<li>Train model from sratch
<li>Score a single event based on user input from command line
<li>Score all events within a given date range, provided by user via command line
</ol>

The flow is:
<ul>
<li>Train a primary model. This will yield an **id_num** that is unique to this model. Navigate to **pickles/classifiers/** and **pickles/features/** to confirm that a classifier and feature pipeline has been created.
<li>Train a secondary model using the above **id_num** as input. This will also create a secondary classifier (**pickles/classifiers/**) and feature pipeline (**pickles/features/**) with the same **id_num**.
<li>Train a tertiary model, using the above **id_num**. Again, the classifier and feature pipeline will be stored. This tertiary model will be the final model used in training events for storage in the DB
</ul>

Additional uses:
<ul>
<li>Scoring individual events using any of the three models
<li>Scoring events within a given date range using any of the three models. 
<li>As above, for Secondary and Tertiary models, take care to use consistent **id_nums**.
</ul>

Examples:

<ol>
<li>Train primary model from scratch

    $ python3 classifiers/primary_train_model

Assuming this model `id_num` is `485` (you can see this by navigating to `pickles/classifiers/`), then

<li>Train a secondary model, using an already trained primary model as input (for the primary classifications).
       
    $ python3 classifiers/secondary_train_model --id_num 485

<li>Train a secondary model, using an already trained primary model as input (for the primary classifications).
       
    $ python3 classifiers/tertiary_train_model --id_num 485

<li>Using secondary classifer to classify all events from March 1, 2017 to March 10, 2017
    
    $ python3 classifiers/secondary_multiple_events --id_num 485 --start_date 2017-03-01 --end_date 2017-03-10

</ol>


