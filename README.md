<H3>NAME: A.J.PRANAV</H3>
<H3>REGISTER NUMBER: 212222230107</H3>
<H3>EX. NO.1</H3>
<H3>DATE</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:

```
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
```
```
df=pd.read_csv("Churn_Modelling.csv")
df
```
![image](https://github.com/user-attachments/assets/672b24a6-a494-45f1-81ea-e437bfc2c192)
```
df.isnull().sum()
```
![image](https://github.com/user-attachments/assets/b5402ff0-f63c-48a6-a3ff-743efaff192b)

```
df.duplicated()
```
![image](https://github.com/user-attachments/assets/685bd32a-2c60-4b44-8d01-65d4df8b2530)

```
print(df['CreditScore'].describe())
```
![image](https://github.com/user-attachments/assets/8c9974da-d84c-4779-923f-758d36443b66)

```
df.info()
```
![image](https://github.com/user-attachments/assets/2c966d0a-56f2-477c-8321-ec7b076b1027)

```
df.drop(['Surname','CustomerId','Geography','Gender'],axis=1,inplace=True)
df
```
![image](https://github.com/user-attachments/assets/5b920055-298d-4aba-a6f0-d1a9d9ad4ad9)

```
scaler=MinMaxScaler()
df=pd.DataFrame(scaler.fit_transform(df))
df
```
![image](https://github.com/user-attachments/assets/dbd118df-a810-4dbf-82b3-d113b547b194)


```
X = df.iloc[:, :-1].values
print(X)
```
![image](https://github.com/user-attachments/assets/bf9c8a80-c712-4c2c-8384-33d712fdff65)

```
y = df.iloc[:,-1].values
print(y)
```
![image](https://github.com/user-attachments/assets/711ff401-09b7-4bc1-adb1-f127506a97fd)

```
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=25)
```

```
print(X_train)
print(len(X_train))
```
![image](https://github.com/user-attachments/assets/5f70c1e0-4b78-41e8-a8ad-8fccafb7609d)


```
print(X_test)
print(len(X_test))
```
![image](https://github.com/user-attachments/assets/32c95eb7-b725-4c69-aceb-313cc299c2cf)










## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


