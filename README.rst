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





