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

    # test if AUC = 0.5 if there are only two experiments falling at (0,0) and (1,1)
    def test_auc_two_cond(self):
        exp = Experiment()

        sd1 = SignalDetection(0, 10, 0, 10)  
        sd2 = SignalDetection(10, 0, 10, 0) 

        exp.add_condition(sd1, label="Condition A")
        exp.add_condition(sd2, label="Condition B")

        auc = exp.compute_auc()

        self.assertAlmostEqual(auc, 0.5, places=3)  

    # test if AUC = 1 if there are three experiments falling at (0,0), (0,1), and (1,1)
    def test_auc_three_cond(self):
        exp = Experiment()

        sd1 = SignalDetection(0, 10, 0, 10)  
        sd2 = SignalDetection(10, 0, 10, 0)  
        sd3 = SignalDetection(10, 0, 0, 10) 

        exp.add_condition(sd1, label="Condition A")
        exp.add_condition(sd2, label="Condition B")
        exp.add_condition(sd3, label="Condition C")

        auc = exp.compute_auc()

        self.assertAlmostEqual(auc, 1.0, places=3)  
        
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

