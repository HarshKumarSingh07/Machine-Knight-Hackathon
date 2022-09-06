# Requirements

1. StreamLit : *pip install streamlit* (To design the web page and deploy the model)
2. Sklearn : *pip install sklearn* (To develop a machine learning algorithm)
3. CatBoost : *pip install catboost* (To use the open source implementation of this strong tree based model)

# This project contains following files

1. *webs.py* : This file contains the python code integrated with HTML/CSS using the tool StreamLit to design an interface for the user so that he/she can use it to predict the rent values. Running this file would simply present us with the local host ID and can be viewed in any browser.

2. *utils.py* : This file contains basic utility code that were used to design the machine learning model.

3. *models.py* : This files contains the treee based machine learning model i.e. CatBoost regression model.

4. *EDA.ipynb* : This file contains the Exploratory data analysis of the given dataset which gives us better insights of the dataset so that we can use it for our analysis

5. *rent.csv* : This file contains the predicted value of the model on the test set

6. *pickle_model* : This contains the saved machine learning model in the binary unreadable format, which we use later to load and make the predictions

7. *home.png* : It is a png file for our website background.

8. *train.csv / test.csv / sample submission.csv* : The trainng, testing and the sample sumbission provided.

# EDA, Featue engineering and Pipeline
On extensively performing the EDA we found that the given dataset had no missing values and had some outliers. Some of the feature vector also showed no relations with the target varaiable. 

Hence in the feature engineering part we used median as the imputing strategy if any missing value is encountered, since medians are robust to outliers. We also experimented with many data transformation techniques such as **yeo-johnson**, **box-cox** transformation **Min-Max** scaling and **Standard Scaling**. Later it was found that **Standard Scaling** performed best. And finally we designed the pipeline so that it becomes easy for our model deployment.

# CatBoost

To decide upon the machine learning model I did extensive probing through various existing SOTA models. We split the training data into a ratio of 80/20 %, the 20% data was used to analyze the predictions and compare the other machine learning models. But after rigrous experimentaion it was found that CatBoost performed better than any other machine learning model which include Ridge regression, Support vector regression, K-nearest-neignours regresion, Random Forest regression, Gaussian regression etc. We also try stacking up a two level model which included XGboost, Random Forest and SVR but still they failed to outperform the CatBoost model. We used the Grid Search algorithm for the hyper-parameter tuning over the 20% training data that was splitted earlier with 10-fold-cross-validation.

The machine learning model that we used for our objective is CatBoost. CatBoost is a relatively new open-source machine learning algorithm, developed in 2017 by a company named Yandex. Some of the salient features of this algorithm are as follows.

1. Great quality without parameter tuning : Reduce time spent on parameter tuning, because CatBoost provides great results with default parameters
2. Categorical features support : Improve your training results with CatBoost that allows you to use non-numeric factors, instead of having to pre-process your data or spend time and effort turning it to numbers.
3. Fast and scalable GPU version : Train your model on a fast implementation of gradient-boosting algorithm for GPU. Use a multi-card configuration for large datasets.
4. Improved accuracy : Reduce overfitting when constructing your models with a novel gradient-boosting scheme.
5. Fast prediction : Apply your trained model quickly and efficiently even to latency-critical tasks using CatBoost's model applier