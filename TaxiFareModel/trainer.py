from TaxiFareModel.data import get_data, clean_data
from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import numpy as np

class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        pipe_distance = make_pipeline(DistanceTransformer(), StandardScaler())
        pipe_time = make_pipeline(TimeFeaturesEncoder(time_column='pickup_datetime'), OneHotEncoder(sparse=False,handle_unknown='ignore'))
        dist_cols = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']
        time_cols = ['pickup_datetime']

        preprocessing = ColumnTransformer([('time', pipe_time, time_cols),
                                      ('distance', pipe_distance, dist_cols)]
                                      , remainder='passthrough')

        pipe_random_forest = Pipeline(steps=[('preprocessing', preprocessing),
                                ('regressor', RandomForestRegressor())])

        self.pipeline = pipe_random_forest
        return self.pipeline

    def run(self, X_train, y_train):
        self.pipeline.fit(X_train,y_train)

    def evaluate(self, X_test, y_test):
        y_pred = self.pipeline.predict(X_test)
        return np.sqrt(((y_pred - y_test)**2).mean())


if __name__ == "__main__":
    df = get_data(nrows=10_000)
    X, y = clean_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    trainer = Trainer(X_train,y_train)
    trainer.set_pipeline()
    trainer.run(X_train,y_train)
    score = trainer.evaluate(X_test,y_test)
    print(score)
    print("GGs")
