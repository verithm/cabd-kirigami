"""Protect soft voting and the physically meaningful log-barrier aggregation."""
from pathlib import Path
import sys
import unittest
import numpy as np
BASE=Path(__file__).resolve().parents[1]
for p in [BASE,BASE/'forward',BASE.parent/'src']:sys.path.insert(0,str(p))
from forward_six.ensemble import soft_vote,mean_regression


class MLPEnsembleTests(unittest.TestCase):
    def test_probability_average_is_not_hard_majority_vote(self):
        members=np.array([[[.49,.51,0]],[[.49,.51,0]],[[.99,.01,0]]])
        labels,probabilities,sd=soft_vote(members)
        self.assertEqual(labels.tolist(),['bi>bi'])
        np.testing.assert_allclose(probabilities,[[1.97/3,1.03/3,0]])
        self.assertEqual(sd.shape,(1,3))

    def test_barrier_averages_learned_log_scale_before_restoration(self):
        pred=mean_regression('barrier',np.array([[[0.]],[[2.]]]),index=[7])
        self.assertEqual(pred.loc[7,'log10_barrier'],1.)
        self.assertEqual(pred.loc[7,'barrier_mJ'],10.)
        self.assertNotEqual(pred.loc[7,'barrier_mJ'],50.5)
        self.assertGreater(pred.loc[7,'barrier_mJ_member_sd'],0)

    def test_invalid_probabilities_are_rejected(self):
        with self.assertRaises(ValueError):soft_vote(np.array([[[np.nan,0,1]],[[0,0,1]]]))
        with self.assertRaises(ValueError):soft_vote(np.array([[[-.1,.1,1]],[[0,0,1]]]))


if __name__=='__main__':unittest.main()
