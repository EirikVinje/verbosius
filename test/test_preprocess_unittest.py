import src.preprocess as preprocess
import unittest


class TestPreprocess(unittest.TestCase):

    def test_clean_text(self):
        """
        Simple test to check that html tags, special characters, large numbers and 
        non-alphanumeric characters are handled correctly in the clean text function
        """

        textdata = ['<p> This is a test sentence with SPECIAL CHARACTERS @#@#$ and numbers 10 1000000000 100 1000 1000000000. </p>', 
                    "<body> Didn't, shouldn't, wouldn't cat's </body>",
                    "<i> 1029384610987246591827631582641"]
        textdata = preprocess.clean_text(textdata)

        self.assertEqual(textdata[0], 'this is a test sentence with special characters and numbers 10', textdata[0])
        self.assertEqual(textdata[1], 'didnt shouldnt wouldnt cats', textdata[1])
        self.assertEqual(textdata[2], "", textdata[2])
    

    def test_lemmatize(self):
        
        """
        Simple test to check if lemmatization is working as expected
        """
        textdata = ["i didnt walk to the store and bought some apples and i didn't buy any oranges"]
        
        tokens, lemmas, token_map = preprocess.lemmatize(textdata)
        
        self.assertEqual(tokens, ['i', 'did', 'nt', 'walk', 'to', 'the', 'store', 'and', 'bought', 'some', 'apples', 'and', 'i', 'did', 'nt', 'buy', 'any', 'oranges'], tokens)
        self.assertEqual(lemmas, ['i', 'did', 'not', 'walk', 'to', 'the', 'store', 'and', 'buy', 'some', 'apple', 'and', 'i', 'did', 'not', 'buy', 'any', 'orange'], tokens)
        self.assertEqual(token_map, [0, 1, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 13, 14, 15, 16], token_map)
    


if __name__ == "__main__":
    
    unittest.main()