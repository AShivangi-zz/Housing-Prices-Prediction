# Model Evaluation and Validation
## Project: Predicting Boston Housing Prices
## Project Overview
In this project, I applied basic machine learning concepts on data collected for housing prices 
in the Boston, Massachusetts area to predict the selling price of a new home. I first 
explored the data to obtain important features and descriptive statistics about the dataset. 
Next, it was properly split into testing and training subsets, and determined a 
suitable performance metric for this problem. I then analyzed performance graphs for a 
learning algorithm with varying parameters and training set sizes which enabled me to pick 
the optimal model that best generalizes for unseen data. Finally, I tested this optimal 
model on a new sample and compare the predicted selling price to your statistics.

## Project Highlights

- Used NumPy to investigate the latent features of a dataset.
- Analysed various learning performance plots for variance and bias.
- Determined the best-guess model for predictions from unseen data.
- Evaluated a model's performance on unseen data using previous data.

## Software and Libraries
This project uses the following software and Python libraries:

- [Python](https://www.python.org/download/releases/3.0/)
- [NumPy](http://www.numpy.org/)
- [pandas](http://pandas.pydata.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [matplotlib](http://matplotlib.org/)

If you do not have Python installed yet, it is highly recommended that you install the [Anaconda](http://continuum.io/downloads) distribution of Python, which already has the above packages and more included. 

This project contains three files:

- `boston_housing.py`: This is the main file.
- `housing.csv`: The project dataset.
- `visuals.py`: This Python script provides supplementary visualizations for the project.

## Running the project
```
python boston_housing.py
```

## Data

The modified Boston housing dataset consists of 489 data points, with each datapoint having 3 features. This dataset is a modified version of the Boston Housing dataset found on the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Housing).

**Features**
1.  `RM`: average number of rooms per dwelling
2. `LSTAT`: percentage of population considered lower status
3. `PTRATIO`: pupil-teacher ratio by town

**Target Variable**
4. `MEDV`: median value of owner-occupied homes