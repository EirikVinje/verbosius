=========
verbosius
=========


ALLIGNING NGRAMS WEIGHTED FROM THE TM:

When alligning the ngrams, e.g we have the text "The cats were dancing all night".
This may be converted to a row of ngrams like : ["the cats", "were", "dancing all night"] depending
on what ngrams the countvectorizer captured. To each feature/ngram, a weight is made from the 
"local weights" method from our thesis report. The feature "the cats" may have the weight 0.5 or any
other number between -1.0 and 1.0. We always want to capture the largest ngram, so if there is a feature
"the" and "cats", we rather take "the cats" if "the cats" is a feature. However, the weight from "the" and "cats" 
is added to weight to "the cats".

When converting to ngrams, we may end up with ngrams beeing intertwined, like : 
["the cats", "cats were dancing", "were dancing all", "all night"]. 

Scenario 1, two bigrams next to eachother:

When we have two bigrams next to eachother where the last word of the first bigram is the first word
in the second bigram, e.g [... "the cats", "cats were" ...], we convert it to a trigram, 
e.g : [... "the cats were" ...]

Scenario 2, intertwined trigram:


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


