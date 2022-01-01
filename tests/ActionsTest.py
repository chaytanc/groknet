import unittest
from actions import Actions


class NetworkTest(unittest.TestCase):

    def setUp(self):
        self.state_size = 3
        init_url = "https://www.google.com"
        self.a = Actions(self.state_size, init_url=init_url)

    def test_see(self):
        pixels = self.a.see()
        # Make sure didn't get blank screenshot / all zeroes
        assert(pixels.any())

    def tearDown(self):
        self.a.end()