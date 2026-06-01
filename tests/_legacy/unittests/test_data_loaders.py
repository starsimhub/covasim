import unittest
import covasim as cv

class TestDataLoaders(unittest.TestCase):
    def test_country_households(self):
        ch = cv.data.get_household_size()

        self.assertTrue(isinstance(ch, dict))
        self.assertGreater(len(ch), 5)

        self.assertTrue(ch['united states of america'] == ch['usa'])
        self.assertTrue(ch['republic of korea'] == ch['korea'])
        self.assertTrue(ch['republic of korea'] == ch['south korea'])

        self.assertTrue(2 <= ch['usa'] <= 5)
        self.assertTrue(1 <= ch['korea'] <= 3)
        self.assertTrue(1 <= ch['germany'] <= 3)
        self.assertTrue(5 <= ch['senegal'] <= 10)


if __name__ == '__main__':
    unittest.TestCase.run = lambda self,*args,**kw: unittest.TestCase.debug(self)
    unittest.main()
