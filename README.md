# Machile Learning used for Stock Trade

This application uses Supervised Machile learning understand new datain changing markets. 

### Data Analyzed and its source
The application reads a data set of 34000 organizations that have recieved funding over the years. The CSV file contains a variety of information about each business including weather or not it became successful. 
These findings can be used to predict weather a business will be successful. The financial institution will use this to decide upon lending to the business.

    emerging_markets_ohlcv.csv
 
### Summary of processes and results
1. Historical data was pulled from csv file and modeled for fitting within the Neural Network set of toolsa
2. Process data for Neural Network Modeling.
3. Compile and Evaluate a Binary Classification Model using a Nerural Network. 
4. Optimize the Neural Network Model using Tensorflow and Keras.
5. Define 2 deep neural network models from the original network model and 2 optimization attempts.
6. Obtain each model's predictions 
7. Predictions are saved in accompanying folder as HDF5 files.

### Results
1. HDF5 files wich include evaluated data using 2 different models
2. Aim is to maximize accuracy of predictions and minimize financial losses if lending occurs

   
---
## Technologies Libraries and Dependencies used
### Python:

    Phyton Version: **3.7.13**

### Sklearn
[sklearn](https://scikit-learn.org/stable/)

### Standard Scaler
[standardscaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)

### Stochastic Gradient Descent- SGD
[stochastic-gradient-descent](https://scikit-learn.org/stable/modules/linear_model.html#stochastic-gradient-descent-sgd) 

### ADA Boost
[ada-boost](https://scikit-learn.org/stable/modules/ensemble.html#adaboost)

### Accuracy Evaluation Metric
[accuracy-evaluation](https://pythonguides.com/adam-optimizer-pytorch/)



## Use case and DEMO
1. Install libraries
      
      pip install sklearn

2. Import dependencies

        import pandas as pd
        import numpy as np
        from pathlib import Path
        import hvplot.pandas
        import matplotlib.pyplot as plt
        from sklearn import svm
        from sklearn.preprocessing import StandardScaler
        from pandas.tseries.offsets import DateOffset
        from sklearn.metrics import classification_report
        from sklearn.linear_model import SGDClassifier
        from sklearn.model_selection import cross_val_score
        from sklearn.datasets import load_iris
        from sklearn.ensemble import AdaBoostClassifier 
      
3. Clone the repository
4. Open file with Jypyter notebooek


## DEMO of tested models

## Training the model
### 2 months training
![2](/images/2.png)
### 3 month training
![3](/images/3.png)
### 6 months 
![6](/images/6.png)
### 12 months 
![12](/images/12.png)

---
## Supervised Learning
## Method 1: SGD Classifier

### SGD Trained 2 Months
![SGD trained 2 months](./images/sgd2.png)

### SGD Trained 12 Months
![SGD trained 12 months](./images/sgd12.png)
---


## Method 2: ADA Boost

---
## Contributors
- Starter code provided by UW Fintech Bootcamp program.
---

## License
Tool is available under an MIT License.

## Aknowledgements
* [Markdown Guide](https://www.markdownguide.org/basic-syntax/#reference-style-links)


