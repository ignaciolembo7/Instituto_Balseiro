## Your own Scikit-learn

NOTE: This is the first __deliverable__ exercise in this exercise set!

Note: Dataset files in this case are in a `pickle` format, a standard python serialization library. To load them, use `pickle`.

1. Implement the SGD training process in `SGDEstimator`, and then implement `LinearRegressor` and `LogisticRegressor`. 
2. In the `explore.ipynb` notebook, explore `dataset_A.pkl` and `dataset_B.pkl` and choose the right model for each dataset. What things did you look to make that decision?
3. In the `train.ipynb` notebook that uses those classes to train respective models for `regression_dataset.csv` and `classification_dataset.csv`. Do a train/test split and measure accuracy for the train and test sets. 
4. Implement the GradientBoostingEstimator class to create compound estimators.
5. Implement the Tree Stump Regressor class.
6. Try to fit a linear regression and a logistic classifier to `dataset_C.csv`. Then, create a new gradient boosting estimator that fits 10, 100 and 1000 tree stumps to it, in the `train.ipynb` notebook. How does accuracy compare? 
7. Rewrite the same notebook using classes from `scikit-learn` in the `sklearn.ipynb` notebook. What differences do you find?
