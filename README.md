<H3>ENTER YOUR NAME : THARUN SRIDHAR</H3>
<H3>ENTER YOUR REGISTER NO.: 212223230230</H3>
<H3>EX. NO.1</H3>
<H3>20-08-2025</H3>
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
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
data = pd.read_csv("Iris.csv")
data
data.head()
X=data.iloc[:,:-1].values
X
y=data.iloc[:,-1].values
y
data.isnull().sum()
data.duplicated()
data.describe()
data = data.drop(['Id','SepalLengthCm','SepalWidthCm'], axis=1)
data.head()
scaler=MinMaxScaler()
# Exclude the 'Species' column before scaling
numerical_data = data.drop('Species', axis=1)
df1 = pd.DataFrame(scaler.fit_transform(numerical_data), columns=numerical_data.columns)
print(df1)
X_train ,X_test ,y_train,y_test=train_test_split(X,y,test_size=0.2)
X_train
X_test
print("Lenght of X_test ",len(X_test))
```


## OUTPUT:
<img width="584" height="346" alt="image" src="https://github.com/user-attachments/assets/c9126bbc-1d1d-4119-b8dc-f16dd40af3bd" />

<img width="620" height="184" alt="image" src="https://github.com/user-attachments/assets/2cc41ba9-47d0-4019-9987-415294870ff9" />

<img width="557" height="546" alt="image" src="https://github.com/user-attachments/assets/4aba48eb-7f32-429e-99f7-9c9b5f90e1b0" />

<img width="669" height="678" alt="image" src="https://github.com/user-attachments/assets/483ce884-7a17-43b7-a1ac-36373ff01aa6" />

<img width="227" height="236" alt="image" src="https://github.com/user-attachments/assets/cdec9702-bbc1-42cf-be6d-9cdcc6f8d924" />

<img width="247" height="381" alt="image" src="https://github.com/user-attachments/assets/66dac154-1baa-4b63-8180-438a95a236ab" />

<img width="646" height="259" alt="image" src="https://github.com/user-attachments/assets/3c94d42e-9b6f-4ef6-80e4-a7c3b53cd0b7" />

<img width="388" height="192" alt="image" src="https://github.com/user-attachments/assets/4056a11a-236a-45a4-b4d2-e1f306076dcf" />

<img width="441" height="223" alt="image" src="https://github.com/user-attachments/assets/31d03a75-fc1f-43c7-94b0-152d6514e959" />

<img width="511" height="726" alt="image" src="https://github.com/user-attachments/assets/afb523ee-fa4f-4c49-bfad-b1f5ba762bdd" />

<img width="418" height="449" alt="image" src="https://github.com/user-attachments/assets/7ac0a9fc-d9c9-465b-9745-1e1f5f9d1d42" />

<img width="262" height="41" alt="image" src="https://github.com/user-attachments/assets/dc093058-8909-4707-9dab-2c50e0d61b99" />



## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


