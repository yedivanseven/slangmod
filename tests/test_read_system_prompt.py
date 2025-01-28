import unittest
from tempfile import NamedTemporaryFile
from slangmod.config.defaults.chat import read_system_prompt


class TestUsage(unittest.TestCase):

    def test_forwards_text(self):
        expected = 'Hello world!'
        actual = read_system_prompt(expected)
        self.assertEqual(expected, actual)

    def test_reads_file(self):
        expected = 'Hello world!'
        with NamedTemporaryFile('w+t') as file:
            file.write(expected)
            file.seek(0)
            actual = read_system_prompt(file.name)
            self.assertEqual(expected, actual)


if __name__ == '__main__':
    unittest.main()
