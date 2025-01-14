import pickle
import unittest
from tokenizers import AddedToken
from slangmod.ml.tokenizers import Special


class TestAddedTokens(unittest.TestCase):

    def test_hashable(self):
        token = AddedToken('[EOS]')
        _ = hash(token)

    def test_default_equality(self):
        one = AddedToken('[EOS]')
        two = AddedToken('[EOS]')
        self.assertEqual(one, two)

    def test_custom_equality(self):
        one = AddedToken(
            '[EOS]',
            single_word=True,
            lstrip=True,
            rstrip=True,
            normalized=True,
            special=True
        )
        two = AddedToken(
            '[EOS]',
            single_word=True,
            lstrip=True,
            rstrip=True,
            normalized=True,
            special=True
        )
        self.assertEqual(one, two)

    def test_default_inequality(self):
        one = AddedToken('[EOS]')
        two = AddedToken('[PAD]')
        self.assertNotEqual(one, two)

    def test_single_word_inequality(self):
        one = AddedToken('[EOS]', single_word=True)
        two = AddedToken('[EOS]', single_word=False)
        self.assertNotEqual(one, two)

    def test_lstrip_inequality(self):
        one = AddedToken('[EOS]', lstrip=True)
        two = AddedToken('[EOS]', lstrip=False)
        self.assertNotEqual(one, two)

    def test_rstrip_inequality(self):
        one = AddedToken('[EOS]', rstrip=True)
        two = AddedToken('[EOS]', rstrip=False)
        self.assertNotEqual(one, two)

    def test_normalized_inequality(self):
        one = AddedToken('[EOS]', normalized=True)
        two = AddedToken('[EOS]', normalized=False)
        self.assertNotEqual(one, two)

    def test_special_inequality(self):
        one = AddedToken('[EOS]', special=True)
        two = AddedToken('[EOS]', special=False)
        self.assertNotEqual(one, two)



class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.pad = AddedToken('[PAD]')
        self.unk = AddedToken('[UNK]')
        self.eos = AddedToken('[EOS]')
        self.special = Special(pad=self.pad, unk=self.unk, eos=self.eos)

    def test_has_unpredictable(self):
        self.assertTrue(hasattr(self.special, 'unpredictable'))

    def test_unpredictable(self):
        self.assertTupleEqual((), self.special.unpredictable)

    def test_has_predictable(self):
        self.assertTrue(hasattr(self.special, 'predictable'))

    def test_predictable(self):
        self.assertTupleEqual((), self.special.predictable)

    def test_has_pad(self):
        self.assertTrue(hasattr(self.special, 'pad'))

    def test_pad(self):
        self.assertIs(self.special.pad, self.pad)

    def test_has_unk(self):
        self.assertTrue(hasattr(self.special, 'unk'))

    def test_unk(self):
        self.assertIs(self.special.unk, self.unk)

    def test_has_eos(self):
        self.assertTrue(hasattr(self.special, 'eos'))

    def test_eos(self):
        self.assertIs(self.special.eos, self.eos)

    def test_has_tokens(self):
        self.assertTrue(hasattr(self.special, 'tokens'))

    def test_tokens(self):
        expected = [self.pad, self.unk, self.eos]
        self.assertListEqual(expected, self.special.tokens)

    def test_has_ids(self):
        self.assertTrue(hasattr(self.special, 'ids'))

    def test_ids(self):
        expected = [0, 1, 2]
        self.assertListEqual(expected, self.special.ids)

    def test_has_contents(self):
        self.assertTrue(hasattr(self.special, 'contents'))

    def test_contents(self):
        expected = [self.pad.content, self.unk.content, self.eos.content]
        self.assertListEqual(expected, self.special.contents)

    def test_has_items(self):
        self.assertTrue(hasattr(self.special, 'items'))

    def test_items(self):
        expected = [(0, self.pad), (1, self.unk), (2, self.eos)]
        self.assertListEqual(expected, self.special.items)

    def test_has_pad_id(self):
        self.assertTrue(hasattr(self.special, 'pad_id'))

    def test_pad_id(self):
        self.assertIsInstance(self.special.pad_id, int)
        self.assertEqual(0, self.special.pad_id)

    def test_has_unk_id(self):
        self.assertTrue(hasattr(self.special, 'unk_id'))

    def test_unk_id(self):
        self.assertIsInstance(self.special.unk_id, int)
        self.assertEqual(1, self.special.unk_id)

    def test_has_eos_id(self):
        self.assertTrue(hasattr(self.special, 'eos_id'))

    def test_eos_id(self):
        self.assertIsInstance(self.special.eos_id, int)
        self.assertEqual(2, self.special.eos_id)

    def test_has_unigram_vocab(self):
        self.assertTrue(hasattr(self.special, 'unigram_vocab'))

    def test_unigram_vocab(self):
        expected = [
            (self.pad.content, 0.0),
            (self.unk.content, 0.0),
            (self.eos.content, 0.0),
        ]
        self.assertListEqual(expected, self.special.unigram_vocab)


class TestCustomAttributes(unittest.TestCase):

    def setUp(self):
        self.pad = AddedToken('[PAD]')
        self.unk = AddedToken('[UNK]')
        self.eos = AddedToken('[EOS]')
        self.cls = AddedToken('[CLS]')
        self.mask = AddedToken('[MASK]')
        self.special = Special(
            [self.cls],
            self.mask,
            pad=self.pad,
            unk=self.unk,
            eos=self.eos
        )

    def test_unpredictable(self):
        self.assertTupleEqual((self.cls,), self.special.unpredictable)

    def test_predictable(self):
        self.assertTupleEqual((self.mask,), self.special.predictable)

    def test_pad(self):
        self.assertIs(self.special.pad, self.pad)

    def test_unk(self):
        self.assertIs(self.special.unk, self.unk)

    def test_eos(self):
        self.assertIs(self.special.eos, self.eos)

    def test_tokens(self):
        expected = [self.pad, self.unk, self.cls, self.eos, self.mask]
        self.assertListEqual(expected, self.special.tokens)

    def test_ids(self):
        expected = [0, 1, 2, 3, 4]
        self.assertListEqual(expected, self.special.ids)

    def test_contents(self):
        expected = [
            self.pad.content,
            self.unk.content,
            self.cls.content,
            self.eos.content,
            self.mask.content
        ]
        self.assertListEqual(expected, self.special.contents)

    def test_items(self):
        expected = [
            (0, self.pad),
            (1, self.unk),
            (2, self.cls),
            (3, self.eos),
            (4, self.mask)
        ]
        self.assertListEqual(expected, self.special.items)

    def test_pad_id(self):
        self.assertIsInstance(self.special.pad_id, int)
        self.assertEqual(0, self.special.pad_id)

    def test_unk_id(self):
        self.assertIsInstance(self.special.unk_id, int)
        self.assertEqual(1, self.special.unk_id)

    def test_eos_id(self):
        self.assertIsInstance(self.special.eos_id, int)
        self.assertEqual(3, self.special.eos_id)

    def test_unigram_vocab(self):
        expected = [
            (self.pad.content, 0.0),
            (self.unk.content, 0.0),
            (self.cls.content, 0.0),
            (self.eos.content, 0.0),
            (self.mask.content, 0.0)
        ]
        self.assertListEqual(expected, self.special.unigram_vocab)



class TestMagic(unittest.TestCase):

    def setUp(self):
        self.pad = AddedToken('[PAD]')
        self.unk = AddedToken('[UNK]')
        self.eos = AddedToken('[EOS]')
        self.cls = AddedToken('[CLS]')
        self.mask = AddedToken('[MASK]')
        self.tokens = [self.pad, self.unk, self.cls, self.eos, self.mask]
        self.special = Special(
            [self.cls],
            self.mask,
            pad=self.pad,
            unk=self.unk,
            eos=self.eos
        )

    def test_default_len(self):
        special = Special(pad=self.pad, unk=self.unk, eos=self.eos)
        self.assertEqual(3, len(special))

    def test_custom_len(self):
        self.assertEqual(5, len(self.special))

    def test_iter(self):
        for i, token in enumerate(self.special):
            self.assertEqual(self.tokens[i], token)

    def test_getitem_int(self):
        for i, _ in enumerate(self.special):
            self.assertEqual(self.tokens[i], self.special[i])

    def test_getitem_slice(self):
        self.assertListEqual(self.tokens[1::2], self.special[1::2])


class TestMisc(unittest.TestCase):

    def setUp(self):
        self.pad = AddedToken('[PAD]')
        self.unk = AddedToken('[UNK]')
        self.eos = AddedToken('[EOS]')
        self.cls = AddedToken('[CLS]')
        self.mask = AddedToken('[MASK]')

    def test_default_repr(self):
        special = Special(pad=self.pad, unk=self.unk, eos=self.eos)
        expected = "Special([PAD], [UNK], [EOS])"
        self.assertEqual(expected, repr(special))

    def test_custom_repr(self):
        special = Special(
            [self.cls],
            self.mask,
            pad=self.pad,
            unk=self.unk,
            eos=self.eos
        )
        expected = "Special([PAD], [UNK], [CLS], [EOS], [MASK])"
        self.assertEqual(expected, repr(special))

    def test_pickle_works(self):
        special = Special(
            [self.cls],
            self.mask,
            pad=self.pad,
            unk=self.unk,
            eos=self.eos
        )
        _ = pickle.loads(pickle.dumps(special))


if __name__ == '__main__':
    unittest.main()
