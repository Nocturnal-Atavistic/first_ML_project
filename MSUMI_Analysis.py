import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, timedelta
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from category_encoders import OneHotEncoder




ticker = ["MSUMI.NS"]
day_msumi = yf.download(ticker, period="max")



print(day_msumi.info(memory_usage="deep"))


print(day_msumi)


hourly_msumi =yf.download(ticker, period="730d", interval="1h")
# msumi_hourly =data.history(period = "730d", interval = "1h")


print(hourly_msumi)



print(hourly_msumi.info(memory_usage="deep"))


# Deleting the extra row header
hrs_msumi= hourly_msumi.droplevel((1), axis=1)
hrs_msumi = hrs_msumi.reset_index()

# hrs_msumi.set_index('Datetime', inplace=True)
print(hrs_msumi)




hrs_msumi.describe()
hrs_msumi["Close"].median()
hrs_msumi[["Close", "High", "Low", "Open"]].plot()



target = "Close"
x = hrs_msumi.drop(columns=target)
y= hrs_msumi[target]

print("x shape is : ", x.shape)
print("y shape is : ", y.shape)



X_train, X_test, y_train, y_test = train_test_split(
    x,y, test_size=0.2, random_state=42
)

print("X_train shape", X_train.shape)
print("y_train shape", y_train.shape)
print("X_test shape", X_test.shape)
print("y_test shape", y_test.shape)




over_sampler = RandomOverSampler(random_state=42)
X_train_over, y_train_over = over_sampler.fit_resample(X_train, y_train)
print("X_train_over shape: ", X_train_over.shape)
X_train_over.head()















