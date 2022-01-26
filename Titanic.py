
## TITANIC SURVIVAL MODEL ##

# VERİ SETİ HİKAYESİ

# Veri seti Titanic gemi kazasında bulunan kişilere ait bilgileri içermektedir.
# Kaggle’da giriş seviyesi için yarışması bulunan bu veriseti;
# 768 gözlem ve 12 değişkenden oluşmaktadır.
# Hedef değişken "Survived" olarak belirtilmiş olup;
# 1 kişinin hayatta kalmasını,
# 0 ise kişinin hayatını kaybetmesini belirtmektedir.

# DEĞİŞKENLER

# Survived – Hayatta Kalma - 1 Hayatta Kaldı, 0 Hayatta Kalamadı
# Pclass – Bilet Sınıfı - 1 = 1. sınıf, 2 = 2.sınıf, 3 = 3.sınıf
# Age – Yaş
# Sibsp – Titanicte’ki kardeş / eş sayısı
# Parch – Titanicte’ki ebeveyn / çocuk sayısı
# Sex – Cinsiyet
# Embarked: – Yolcunun gemiye biniş yaptığı liman - (C = Cherbourg, Q = Queenstown, S = Southampton
# Fare – Bilet ücreti
# Cabin: Kabin numarası

#importing the necessary libraries

import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, plot_roc_curve
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
warnings.simplefilter(action="ignore")

df = pd.read_csv("datasets/titanic.csv")
df.head()
df.shape

# Exploratory Data Analysis

df.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)

    print("##################### Types #####################")
    print(dataframe.dtypes)

    print("##################### Head #####################")
    print(dataframe.head(head))

    print("##################### Tail #####################")
    print(dataframe.tail(head))

    print("##################### NA #####################")
    print(dataframe.isnull().sum())

    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)

# Numerical and Categorical Variables

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optional
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat

    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]

    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car
cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col != "PassengerId"]
cat_cols = [col for col in cat_cols if col != "Survived"]

# Categorical Variable Analysis

def cat_summary(dataframe, col_name, plot=False):

    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


cat_summary(df,"Survived", True)

for col in cat_cols:
    cat_summary(df, col)

# Numerical Variable Analysis

def num_summary(dataframe, numerical_col, plot=False):

    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]

    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()


num_summary(df, "Age")

for col in num_cols:
    num_summary(df, col, plot=True)

# Target Variable Analysis

def target_summary(dataframe, target, categorical_col, numerical_col):
    for var in dataframe:
        if var in cat_cols:
             print(pd.DataFrame({"RATIO":100*dataframe[var].value_counts(),
                                "TARGET_MEAN": dataframe.groupby(var)[target].mean()}), end="\n\n\n")
        if var in num_cols:
            print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(var)[target].mean()}), end="\n\n\n")

target_summary(df, "Survived", cat_cols, num_cols)

sns.countplot(x="Survived", data=df)
plt.show()

# DATA PRE-PROCESSING

# Outliers

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    print(col, check_outlier(df, col))

def grab_outliers(dataframe, col_name, index=False):
    low,up= outlier_thresholds(dataframe,col_name)
    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

grab_outliers(df,"Fare",True)


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if col not in "PassengerId"]

df.shape

for col in num_cols:
    replace_with_thresholds(df, col)

grab_outliers(df,"Fare",True)

# Missing Values

df.isnull().values.any()
df.isnull().sum()

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns


missing_values_table(df)
missing_values_table(df, True)

df["Age"] = df["Age"].fillna(df.groupby("Sex")["Age"].transform("median"))
df = df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0)



missing_values_table(df)

# New Feature

df["New_Cabin_Bool"] = df["Cabin"].notnull().astype('int')

df["New_Name_Count"] = df["Name"].str.len()

df["New_Name_Word_Count"] = df["Name"].apply(lambda x: len(str(x).split(" ")))

df["New_Name_Dr"] = df["Name"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))

df['New_Title'] = df.Name.str.extract(' ([A-Za-z]+)\. ', expand=False)

df["New_Family_Size"] = df["SibSp"] + df["Parch"] + 1

df["New_Age_Pclass"] = df["Age"] * df["Pclass"]

df.loc[((df['SibSp'] + df['Parch']) > 0), "New_Is_Alone"] = "NO"
df.loc[((df['SibSp'] + df['Parch']) == 0), "New_Is_Alone"] = "YES"

df.loc[(df['Age'] < 18), 'New_Age_Cat'] = 'young'
df.loc[(df['Age'] >= 18) & (df['Age'] < 56), 'New_Age_Cat'] = 'mature'
df.loc[(df['Age'] >= 56), 'New_Age_Cat'] = 'senior'

df.loc[(df['Sex'] == 'male') & (df['Age'] <= 21), 'New_Sex_Cat'] = 'youngmale'
df.loc[(df['Sex'] == 'male') & ((df['Age'] > 21) & (df['Age']) <= 50), 'New_Sex_Cat'] = 'maturemale'
df.loc[(df['Sex'] == 'male') & (df['Age'] > 50), 'New_Sex_Cat'] = 'seniormale'
df.loc[(df['Sex'] == 'female') & (df['Age'] <= 21), 'New_Sex_Cat'] = 'youngfemale'
df.loc[(df['Sex'] == 'female') & ((df['Age'] > 21) & (df['Age']) <= 50), 'New_Sex_Cat'] = 'maturefemale'
df.loc[(df['Sex'] == 'female') & (df['Age'] > 50), 'New_Sex_Cat'] = 'seniorfemale'

df.columns = [col.upper() for col in df.columns]

remove_cols = ["TICKET", "NAME"]
df.drop(remove_cols, inplace=True, axis=1)


##################
# Rare Analyser, Rare Encoding
##################
# Rare Analyser

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

cat_cols, num_cols, cat_but_car = grab_col_names(df)
rare_analyser(df, "SURVIVED", cat_cols)

# Rare encoder
def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df

df = rare_encoder(df, 0.01)

df.head()

##################
# Label Encoding & One-Hot Encoding
##################

cat_cols, num_cols, cat_but_car = grab_col_names(df)

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtypes == "O"
               and len(df[col].unique()) == 2]

binary_cols

for col in binary_cols:
    label_encoder(df, col)

df.head()

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe
ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() >= 2 and col not in "SURVIVED"]
df = one_hot_encoder(df, ohe_cols, True)
df.head()

cat_cols, num_cols, cat_but_car = grab_col_names(df)


useless_cols = [col for col in df.columns if df[col].nunique() == 2 and
                (df[col].value_counts() / len(df) < 0.01).any(axis=None)]

## Standard Scaler

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
df[num_cols].head()
df.head()

#############################################
# MODEL
#############################################

y = df["SURVIVED"]
X = df.drop(["PASSENGERID","SURVIVED"], axis=1)

log_model = LogisticRegression().fit(X, y)

log_model.intercept_
log_model.coef_

# Tahmin
y_pred = log_model.predict(X)

y_pred[0:10]
y[0:10]

# Confusion Matrix
def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy Score: {0}'.format(acc), size=10)
    plt.show()

plot_confusion_matrix(y, y_pred)

print(classification_report(y, y_pred))

#               precision    recall  f1-score   support
#            0       0.85      0.88      0.87       549
#            1       0.80      0.76      0.78       342
#     accuracy                           0.83       891
#    macro avg       0.83      0.82      0.82       891
# weighted avg       0.83      0.83      0.83       891


# ROC AUC
y_prob = log_model.predict_proba(X)[:, 1]
roc_auc_score(y, y_prob)

## roc_auc_score : 0.88

######################################################
# Model Validation: Holdout
######################################################


# Holdout Yöntemi

# Veri setinin train-test olarak ayrılması:
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.20, random_state=17)


# Modelin train setine kurulması:
log_model = LogisticRegression().fit(X_train, y_train)

# Test setinin modele sorulması:
y_pred = log_model.predict(X_test)

# AUC Score için y_prob (1. sınıfa ait olma olasılıkları)
y_prob = log_model.predict_proba(X_test)[:, 1]


# Classification report
print(classification_report(y_test, y_pred))

#               precision    recall  f1-score   support
#            0       0.79      0.86      0.82       106
#            1       0.77      0.67      0.72        73
#     accuracy                           0.78       179
#    macro avg       0.78      0.76      0.77       179
# weighted avg       0.78      0.78      0.78       179

# ROC Curve
plot_roc_curve(log_model, X_test, y_test)
plt.title('ROC Curve')
plt.plot([0, 1], [0, 1], 'r--')
plt.show()

roc_auc_score(y_test, y_prob)



######################################################
# Model Validation: 10-Fold Cross Validation
######################################################

y = df["SURVIVED"]
X = df.drop(["SURVIVED","PASSENGERID"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)

log_model = LogisticRegression().fit(X_train, y_train)

cv_results = cross_validate(log_model,
                            X, y,
                            cv=5,
                            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])



cv_results['test_accuracy'].mean()
# Accuracy: 0.8260

cv_results['test_precision'].mean()
# Precision: 0.7920

cv_results['test_recall'].mean()
# Recall: 0.7424

cv_results['test_f1'].mean()
# F1-score: 0.7651

cv_results['test_roc_auc'].mean()
# AUC: 0.8622


# Prediction for a new observation

X.columns

random_user = X.sample(1, random_state=44)

log_model.predict(random_user)