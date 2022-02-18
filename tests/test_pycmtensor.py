#!/usr/bin/env python

"""Tests for `pycmtensor` package."""


import unittest

import pandas as pd

import pycmtensor as cmt


class TestPycmtensor(unittest.TestCase):
    """Tests for `pycmtensor` package."""

    def setUp(self):
        """Set up test fixtures, if any."""
        pass

    def tearDown(self):
        """Tear down test fixtures, if any."""
        pass

    def test_simple(self):
        swissmetro = pd.read_csv("../data/swissmetro.dat", sep="\t")
        db = cmt.Database("swissmetro", swissmetro, choiceVar="CHOICE")
