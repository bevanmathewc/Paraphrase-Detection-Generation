import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


class ReportGenerator:

    def __init__(self, experiment_file="../logs/experiments.xlsx"):
        self.experiment_file = experiment_file
        self.df = pd.read_excel(experiment_file)

    def get_completed_experiments(self):
        return self.df[self.df["Status"] == "Completed"]

    def save_csv(self, df, filename):

        output = Path("../results/metrics")

        output.mkdir(parents=True, exist_ok=True)

        df.to_csv(output / filename, index=False)

        return df
    
    def generate_model_comparison(self):

        comparison = self.get_completed_experiments()[[
            "Model",
            "Dataset",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "ROC-AUC",
            "Train Time",
            "Inference Time"
        ]]

        comparison = comparison.sort_values(
            ["Model", "Dataset"]
        )

        self.save_csv(
            comparison,
            "model_comparison.csv"
        )

        return comparison

    