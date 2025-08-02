from sklearn.ensemble import IsolationForest
import numpy as np

class OutlierDetector:
    def detect(self, df, contamination=0.05):
        """Identify outliers using Isolation Forest"""
        clf = IsolationForest(contamination=contamination)
        outliers = clf.fit_predict(df.select_dtypes(include=['number']))
        return outliers == -1
