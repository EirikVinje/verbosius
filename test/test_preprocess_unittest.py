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
        clean_texts = ["i didnt see this movie", "i have seen cats and dogs"]

        texts, tokens, lemmas, labels = preprocess.lemmatize(clean_texts, [0, 1])

        self.assertEqual(lemmas[0], ['i', 'do', 'not', 'see', 'this', 'movie'], lemmas[0])
        self.assertEqual(lemmas[1], ['i', 'have', 'see', 'cat', 'and', 'dog'], lemmas[1])

        self.assertEqual(tokens[0], ['i', 'did', 'nt', 'see', 'this', 'movie'], tokens[0])
        self.assertEqual(tokens[1], ['i', 'have', 'seen', 'cats', 'and', 'dogs'], tokens[1])

        self.assertEqual(texts[0], ["i", "didnt", "see", "this", "movie"], texts[0])
        self.assertEqual(texts[1], ["i", "have", "seen", "cats", "and", "dogs"], texts[1])


    def test_map_tokens(self):

        """
        Simple test to check if mapping tokens to lemmas is working as expected
        """

        tokens = ["i", "did", "nt", "see", "this", "movie"]
        stext = ["i", "didnt", "see", "this", "movie"]

        ids = preprocess.map_tokens(stext, tokens)

        self.assertEqual(ids, [0, 1, 1, 2, 3, 4], ids)

        tokens = ["i", "l", "o", "v", "e", "c", "a", "t", "s"]
        stext = ["i", "love", "cats"]

        ids = preprocess.map_tokens(stext, tokens)

        self.assertEqual(ids, [0, 1, 1, 1, 1, 2, 2, 2, 2], ids)
        


if __name__ == "__main__":
    
    unittest.main()