# Regular Grammar induction Using Recurrent Neural Network

Code for training and evaluation of the model from ["Regular Grammar induction Using Recurrent Neural Network"](https://arxiv.org/abs/...).  

## Run the program end-to-end

To start training the model for PTB grammar, first download the dataset, available at <https://web.archive.org/web/19970614160127/http://www.cis.upenn.edu:80/~treebank/>, and extract it into the `./Penn_Treebank` directory.

Then use the following command:

```
python main.py
```

The following packages are required:

* Python 3.5
* Tensorflow 1.1
* Scipy
* Sklearn
* Matplotlib
* Pydot


The following parameters can be configured:

```
[Data]
grammatical_source - Should be 'regex', 'ptb' for Penn Treebank, 'phonology' or 'phonology_plurals' (wasn't mentioned in the paper)
ungrammatical_by_trans - Boolean, allows transformations for creating ungrammatical sentences
size_train - Amount of data to be generated for the training set 
size_validation - Amount of data to be generated for the validation set
size_test - Amount of data to be generated for the test set

[Regex]
regex - If grammatical_source was chosen to run 'regex' then state here the regex in synthesizedformal language
alphabet - The alphabet of the regex
max_len = Max sentence length to be generated
min_len = Min sentence length to be generated

[PTB]
alphabet - If grammatical_source was chosen to run 'ptb' then state here the requested alphabet
filter_alphabet - Boolean, choose whether to filter ungrammatical sentences created with the grammaticals
use_orig_sent - Boolean, choose whether to use original PTB data, or synthesized data (created by concatenation of random PTB sentences)

[RNN]
NUM_EPOCHS - the number of epochs to train the RNN
state_size - the dimension of the state vector

[ClusteringModel]
use_model - Boolean, choose whether to use clustering model on the graph of states or not
model - choose 'k_means' or 'meanshift' model for clustering

[Misc]
output_path - path to output graphs and visualisations
```

Parameters can be set by changing their value in the config file - config.cfg.
