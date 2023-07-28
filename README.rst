=========
verbosius
=========

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





BATCHING DATA:

In batch data we've set it so that batchsizes for train and test can be configured
seperately. If no test sizes are set a 80/20 split is used, unless there is a testset
that is to be used, in that case the batchsizes are copied form the training settings.

The function requires reading all the data at once, then batching it. 
After doing so the data is batched and saved balanced over the amount of classes we have
in our data.


STAGER(S):

For all the different stagers we've opted for allowing the user to define all file storing
loactions. This requires absolute paths to be given at all times. This is done to allow
for the user to store the data where they want, and not be forced to store it in a specific
location.


