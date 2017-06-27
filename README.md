# Event Classifier
## Usage - Command line utility

### Help: `$ python3 event_classifier -h`

### 5 ways to use

<ol>
<li>Retrain primary model from scratch

    $ python3 event_classifier  --level primary --retrain T

<li>Load a trained primary model and test it against a dataset.
       
    $ python3 event_classifier --level primary --retrain F \
                               --load_clf classifier/_primary_210.pkl

<li>Retrain a secondary model from scratch

This requires the output of a primary model to provide the primary features as input. Need to load a primary model (step 2) and then use that model to score

    $ python3 event_classifier --level secondary --retrain T \
                               --load_clf classifier/_primary_210.pkl

<li>Load a secondary model

This model uses the output of a primary model as features to test on

    $ python3 event_classifier --level secondary --retrain F \
                               --load_clf classifier/_primary_210.pkl classifier/_secondary_210.pkl


<li>Run a secondary model on a single event

Load a primary and secondary model (step 4) and score single event

    $ python3 event_classifier --level secondary --retrain F \
                               --load_clf classifier/_primary_210.pkl classifier/_secondary_210.pkl \
                               --event_ids 2826859
</ol


### In general:

1.  ``` --level``` can be `primary` or `secondary`
2. `--retrain` can be `T`(True) or `F` (False)
3. `--load_clf` must be a classifier in the `classifier/` directory. Filename will have `_classifier_120.pkl` ending, where the `120` some randomly generated number, doesn't mean anything except that the primary and secondary classifier must have the same number.
4. `--event_ids` must be some valid event id. Will throw an error if not. Currently only support single event classifications, but eventually will be chunks of events.

