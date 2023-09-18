=========
verbosius
=========

SETUP:

To setup env for develpoment:
1. create conda env w/python=3.10
2. run pip install -r requirements.text
3. run pip install -e .
4. run python -m spacy download en_core_web_sm
5. should be ablet o run ./run.sh after this 

PREPROCESSING:

The preprocessing module is used to preprocess the data. It is used to create the
data in a format that is usable for the rest of the program. Checkpoints that the 
data is passed through are:

1. Reading the data from the source

2. Batching it to a specific size and number

3. Cleaning the text like removing punctuation, special characters, etc. which
    results in the text being on lowercase and with words seperated by spaces.

4. Lemmatizing the text, which results in the text being transformed to lemma form, e.g the word 
    'running' becomes 'run' and the word 'havent' becomes 'have' 'not'.

5. Making a map of the words in a text, so we are able to backtrack and find e.g the two words 
    [..."have", "not"....] belongs to the same word, namely "havent". 

6. Staging the data, which means that the data is stored in a specific format, e.g. a pickle file
    or a csv file. This is done to make it easier to load the data later on.


TRAININGDATA:





CHUNKING DATA:

In chunk data we've set it so that chunksizes for train and test can be configured
seperately. If no test sizes are set a 80/20 split is used, unless there is a testset
that is to be used, in that case the batchsizes are copied form the training settings.

The function requires reading all the data at once, then batching it. 
After doing so the data is batched and saved balanced over the amount of classes we have
in our data.

The function also has the ability to  extract validation data from the training data. This is done by percent split if no
val set is available. 

The chunks can also be supersampled, i.e. if we have 25000 datapoints but want 4 chunks of 8000 samples (32k total)
each chunk receives sampels from the otehr chunks randomly til they have size of 8000. Class balance is kept after this.
When train/val split happens it is taken from the training chunks, i.e. a 80/20 split will give trianing chunk of size 6400 etc...

STAGER(S):

For all the different stagers we've opted for allowing the user to define all file storing
loactions. This requires absolute paths to be given at all times. This is done to allow
for the user to store the data where they want, and not be forced to store it in a specific
location.


