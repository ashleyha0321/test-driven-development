import numpy as np 
import scipy
import matplotlib.pyplot as plt
from SignalDetection import SignalDetection 

# class was constructed with the assistance of ChatGPT

class Experiment:
    def __init__(self):
        self.conditions = []
    
    def add_condition(self, sdt_obj: SignalDetection, label: str = None) -> None:
        self.conditions.append((sdt_obj, label))

    def sorted_roc_points(self) -> tuple[list[float], list[float]]:
        if not self.conditions:
            raise ValueError("Value Error: No conditions present")
        
        false_alarm_rates = [sdt_obj.false_alarm_rate() for sdt_obj, _ in self.conditions]
        hit_rates = [sdt_obj.hit_rate() for sdt_obj, _ in self.conditions]

        # sorting false alarm rates
        sorted_pairs = sorted(zip(false_alarm_rates, hit_rates))
        sorted_false_alarms, sorted_hits = zip(*sorted_pairs)

        return list(sorted_false_alarms), list(sorted_hits)

    def compute_auc(self) -> float:
        if not self.conditions:
            raise ValueError("No conditions present to compute AUC.")

        false_alarm_rates, hit_rates = self.sorted_roc_points()

        # calculate auc value using trapezoidal rule
        auc = np.trapz(hit_rates, false_alarm_rates)
        return auc

    def plot_roc_curve(self, show_plot: bool = True):
         if not self.conditions:
            raise ValueError("No conditions present to plot ROC curve.")

         false_alarm_rates, hit_rates = self.sorted_roc_points()

         plt.figure(figsize=(6, 6))
         plt.plot(false_alarm_rates, hit_rates, marker='o', linestyle='-', label="ROC Curve")
         plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Chance (AUC=0.5)")
        
         auc = self.compute_auc()
         plt.title(f"ROC Curve (AUC = {auc:.3f})")
         plt.xlabel("False Alarm Rate")
         plt.ylabel("Hit Rate")
         plt.legend()
         plt.grid()

         if show_plot == True:
             plt.show()










