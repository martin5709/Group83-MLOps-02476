import group83_mlops.train as tr
import torch
import pytest
import group83_mlops.data

def test_hydra():
    """Does a simple test of the train hydra function, to make sure it can run without any errors, and that it runs for 1 epoch"""
    tr.train_hydra(quick_test=True)