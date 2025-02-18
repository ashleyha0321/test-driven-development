import unittest
from Experiment import Experiment
from SignalDetection import SignalDetection

class TestExperiment(unittest.TestCase):
   
    def test_add_condition(self):
        exp = Experiment()
        sd = SignalDetection(40, 10, 20, 30)
        exp.add_condition(sd, label="Condition A")

        self.assertEqual(len(exp.conditions), 1)
        self.assertEqual(exp.conditions[0][1], "Condition A")
        self.assertIs(exp.conditions[0][0], sd) 

    def test_roc_points(self):
        exp = Experiment()
        exp.add_condition(SignalDetection(50, 10, 20, 40), label="Condition A")  
        exp.add_condition(SignalDetection(40, 15, 30, 35), label="Condition B")  

        false_alarms, hits = exp.sorted_roc_points()

        # checking that the false alarms are sorted
        self.assertTrue(false_alarms[0] <= false_alarms[1])

    def test_compute_auc(self):
        exp = Experiment()
        sd1 = SignalDetection(50, 10, 20, 40)
        sd2 = SignalDetection(40, 15, 30, 35)

        exp.add_condition(sd1, label="Condition A")
        exp.add_condition(sd2, label="Condition B")

        auc = exp.compute_auc()
        self.assertAlmostEqual(auc, 0.5, places=3)  

    # test AUC with no conditions
    def test_compute_auc_no_cond(self):
        exp = Experiment()
        with self.assertRaises(ValueError):
            exp.compute_auc()

    # test sorted ROC with no conditions present
    def test_sorted_roc_no_cond(self):
        exp = Experiment()
        with self.assertRaises(ValueError):
            exp.sorted_roc_points()

    # test ROC plot with no conditions present
    def test_plot_roc_no_cond(self):
        exp = Experiment()
        with self.assertRaises(ValueError):
            exp.plot_roc_curve()

    # test AUC computation with multiple conditions
    def test_multi_cond_auc(self):
        exp = Experiment()
        exp.add_condition(SignalDetection(50, 10, 20, 40), label="Condition A") 
        exp.add_condition(SignalDetection(40, 15, 30, 35), label="Condition B") 
        exp.add_condition(SignalDetection(60, 5, 10, 45), label="Condition C")

        auc = exp.compute_auc()
        self.assertTrue(0.5 <= auc <= 1.0)  

    # test experiment with identical conditions
    def test_identical_cond_experiment(self):
        exp = Experiment()
        sd = SignalDetection(50, 10, 20, 40)  
        exp.add_condition(sd, label="Same1")
        exp.add_condition(sd, label="Same2")
        
        false_alarms, hits = exp.sorted_roc_points()
        self.assertEqual(false_alarms[0], false_alarms[1])
        self.assertEqual(hits[0], hits[1])

if __name__ == '__main__':
    unittest.main()

