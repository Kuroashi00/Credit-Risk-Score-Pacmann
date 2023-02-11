#!/usr/bin/env python
# coding: utf-8

# # Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# # Importing Data

# In[2]:


def importData(filename_1, filename_2):
    """
    Fungsi untuk import data dan hapus duplikat
    :param filename <str> : nama file input (.csv)
    :return df <pandas.df>: sampel data
    """
    
    # merge and print data
    credit = pd.read_csv('credit_record.csv')
    application = pd.read_csv('application_record.csv')
    data = pd.merge(credit, application, how='inner')
    print("data asli         = ", data.shape, "- obeservasi, kolom")
    
    # drop duplicates
    data = data.drop_duplicates()
    print("data setelah drop = ", data.shape, "- obeservasi, kolom") 
    
    return data

# (filename) adalah argumen
# Argumen adalah sebuah variable. 
# Jika fungsi tsb diberi argumen filename1 = "credit_data.csv" dan filename2 = "application_record.csv", 
# maka semua variabel 'filename 1 dan 2' di dalam fungsi 
# akan berubah menjadi gabungan dari "credit_data.csv" dan application_record.csv


# In[3]:


#input
file_credit = "credit_record.csv"
file_application = "application_record.csv"

#panggil fungsi
data = importData(filename_1 = file_credit, filename_2 = file_application)


# In[4]:


# Data Preprocessing
# Fitur y adalah output variable dan sisanya adalah input
data.head()


# In[5]:


data.STATUS.value_counts()


# # Define Target and Feature

# In[6]:


"""
Membagi target menjadi 2 kelompok

Data yang termasuk bad status
0: 1-29 days past due 
1: 30-59 days past due 
2: 60-89 days overdue 
3: 90-119 days overdue 
4: 120-149 days overdue 
5: Overdue or bad debts, write-offs for more than 150 days 

Data yang termasuk good status
X: No loan for the month
C: paid off that month

"""

bad_status = ['0','1','2','3','4','5']

# membuat kolom baru yang berisi flag yang mendindikasikan seseorang adalah bad atau good

data['bad_flag'] = np.where(data['STATUS'].isin(bad_status), 1, 0)
data.drop('STATUS', axis=1, inplace=True)


# In[7]:


data.bad_flag.value_counts(normalize=True)


# In[8]:


def extractIO(data,
             output_column_name):
    """
    Fungsi untuk memisahkan output dan input
    :param data <pandas.df> : data seluruh sample
    :param output_column_name <str> : nama kolom output
    :return input_data: <pandas dataframe> data input
    :return output_data: <pandas series> data output
    """
    
    # drop data
    #data = data.drop(columns = column_to_drop)
    output_data = data[output_column_name]
    input_data = data.drop(output_column_name,
                          axis = 1)
    
    return input_data, output_data

# (data, output_column_name) adalah argumen
# Argumen adalah sebuah variable. 
# Jika fungsi tsb. diberi argumen data = credit_data dan application_data, 
# maka semua variabel 'data' di dalam fungsi akan berubah menjadi credit_data application_data


# # Cleaning, Preprocessing, Feature Engineering

# In[9]:


#column_to_drop = ["Unnamed: 0"]
output_column_name = ['bad_flag']

X, y = extractIO(data = data,
                 output_column_name = output_column_name)


# In[10]:


#sanitiCheck
X


# In[11]:


data[data.bad_flag==1]


# In[12]:


#check proporsi data melalu value_counts

for i in X.columns:
    print(i, ':', len(X[i].value_counts()))


# In[13]:


# Terdapat data yang hanya berisi 1 nilai
data.FLAG_MOBIL.value_counts()


# In[14]:


data.MONTHS_BALANCE.value_counts()


# In[15]:


# Karena kolom FLAG_MOBIL hanya berisi satu macam data, maka harus didrop

data = data.drop('FLAG_MOBIL', axis=1)


# In[16]:


# redefine the input and output after change some values of class in repayment and education
X, y = extractIO(data = data,
                          output_column_name = output_column_name)


# In[17]:


#sanity check
X


# In[18]:


#sanity check
y


# In[20]:


#Sanity Check hasil splitting
print(X_train.shape)
print(X_test.shape)


# In[21]:


# Mengecek missing value

X_train.isnull().sum()

# Output: nama variabel, jumlah null value.
# Ada 192058 data yang kosong pada occupation_type column 


# In[22]:


#_get_numeric_data() hanya akan mengambil column berisikan integer dan float
X_train_numerical = X_train._get_numeric_data() 
X_train_numerical.head()


# In[23]:


X_train


# In[25]:


X_train.columns


# In[26]:


# Buat kolom categoric dan numeric 
numerical_column = ['ID', 'MONTHS_BALANCE',
                    'CNT_CHILDREN', 'AMT_INCOME_TOTAL',
                    'DAYS_BIRTH', 'DAYS_EMPLOYED', 'FLAG_WORK_PHONE',
                   'FLAG_PHONE', 'FLAG_EMAIL', 'CNT_FAM_MEMBERS']
categorical_column = ["CODE_GENDER", "FLAG_OWN_CAR", "CODE_GENDER", "FLAG_OWN_REALTY",
                    "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE",
                    "OCCUPATION_TYPE"]


# In[27]:


# Periksa lagi missing value
categorical_data = X_train[categorical_column]
categorical_data.isnull().sum()


# In[28]:


def categoricalImputation(data, categorical_column):
    """
    Fungsi untuk melakukan imputasi data kategorik
    :param data: <pandas dataframe> sample data input
    :param categorical_column: <list> list kolom kategorikal data
    :return categorical_data: <pandas dataframe> data kategorikal
    """
    # seleksi data
    categorical_data = data[categorical_column]

    # lakukan imputasi
    categorical_data = categorical_data.fillna(value="KOSONG")

    return categorical_data


# In[29]:


X_train_categorical = categoricalImputation(data = X_train,
                                            categorical_column = categorical_column)

X_train_categorical.isnull().sum()


# In[30]:


categorical_ohe = pd.get_dummies(X_train_categorical)
categorical_ohe.head(2)


# In[31]:


def extractCategorical(data, categorical_column):
    """
    Fungsi untuk ekstrak data kategorikal dengan One Hot Encoding
    :param data: <pandas dataframe> data sample
    :param categorical_column: <list> list kolom kategorik
    :return categorical_ohe: <pandas dataframe> data sample dengan ohe
    """
    data_categorical = categoricalImputation(data = data,
                                             categorical_column = categorical_column)
    categorical_ohe = pd.get_dummies(data_categorical)

    return categorical_ohe


# In[32]:


X_train_categorical_ohe = extractCategorical(data = X_train,
                                             categorical_column = categorical_column)


# In[33]:


X_train_categorical_ohe.head()


# In[34]:


# Simpan kolom OHE untuk diimplementasikan dalam testing data

ohe_columns = X_train_categorical_ohe.columns


# In[35]:


ohe_columns


# In[36]:


# Join data Numerical dan Categorical

X_train_concat = pd.concat([X_train_numerical,
                            X_train_categorical_ohe],
                           axis = 1)


# In[37]:


X_train_categorical_ohe


# In[38]:


X_train_concat.shape


# In[39]:


X_train_concat.isnull().any()


# In[40]:


from sklearn.preprocessing import StandardScaler

# Buat fungsi normalisasi

def standardizerData(data):
    """
    Fungsi untuk melakukan standarisasi data
    :param data: <pandas dataframe> sampel data
    :return standardized_data: <pandas dataframe> sampel data standard
    :return standardizer: method untuk standardisasi data
    """
    data_columns = data.columns
    data_index = data.index
    
    # buat fit standardizer
    standardizer = StandardScaler()
    standardizer.fit(data)
    
    # transform_data
    standardized_data_raw = standardizer.transform(data)
    standardized_data = pd.DataFrame(standardized_data_raw)
    standardized_data.columns = data_columns
    standardized_data.index = data_index
    
    return standardized_data, standardizer


# In[41]:


X_train_clean, standardizer = standardizerData(data = X_train_concat)
X_train_clean


# In[42]:


# Menentukan baseline

data['bad_flag'].value_counts(normalize=True)*100


# In[43]:


# baseline akurasi = 61%


# # Training & Modeling

# In[19]:


# Import train-test splitting library dari sklearn (scikit learn)
from sklearn.model_selection import train_test_split

"""
1. `X` adalah input
2. `y` adalah output (target)
3. `test_size` adalah seberapa besar proporsi data test dari keseluruhan data. Contoh `test_size = 0.2` artinya data test akan berisi 20% data.
4. `random_state` adalah kunci untuk random. Harus di-setting sama. Misal `random_state = 123`.
5. Output:
   - `X_train` = input dari data training
   - `X_test` = input dari data testing
   - `y_train` = output dari data training
   - `y_test` = output dari data testing
6. Urutan outputnya: `X_train, X_test, y_train, y_test`. Tidak boleh terbalik
"""

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size = 0.2,
                                                    random_state = 2
                                                    )


# In[44]:


# Import dari sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# In[45]:


# Model Random Forest Classifier
random_forest = RandomForestClassifier(random_state = 123)
random_forest.fit(X_train_clean, y_train)

# Model Random Forest Classifier 1
# Mari kita ubah hyperparameter dari random forest --> n_estimator
# Tambahkan n_estimator = 350

random_forest_1 = RandomForestClassifier(random_state = 123,
                                         n_estimators = 350)
random_forest_1.fit(X_train_clean, y_train)

predicted_rf = pd.DataFrame(random_forest.predict(X_train_clean))
predicted_rf.head()


# In[46]:


predicted_rf = pd.DataFrame(random_forest.predict(X_train_clean))
predicted_rf.value_counts()


# In[47]:


random_forest.score(X_train_clean, y_train)


# In[48]:


random_forest_1.score(X_train_clean, y_train)


# In[54]:


def extractTest(data,
                numerical_column, categorical_column, ohe_column, standardizer):
    """
    Fungsi untuk mengekstrak & membersihkan test data 
    :param data: <pandas dataframe> sampel data test
    :param numerical_column: <list> kolom numerik
    :param categorical_column: <list> kolom kategorik
    :param ohe_column: <list> kolom one-hot-encoding dari data kategorik
    :param imputer_numerical: <sklearn method> imputer data numerik
    :param standardizer: <sklearn method> standardizer data
    :return cleaned_data: <pandas dataframe> data final
    """
    # Filter data
    numerical_data = data[numerical_column]
    categorical_data = data[categorical_column]


    # Proses data kategorik
    categorical_data = categorical_data.fillna(value="KOSONG")
    categorical_data.index = data.index
    categorical_data = pd.get_dummies(categorical_data)
    categorical_data.reindex(index = categorical_data.index, 
                             columns = ohe_column)

    # Gabungkan data
    concat_data = pd.concat([numerical_data, categorical_data],
                             axis = 1)
    cleaned_data = pd.DataFrame(standardizer.transform(concat_data))
    cleaned_data.columns = concat_data.columns

    return cleaned_data

def testPrediction(X_test, y_test, classifier, compute_score):
    """
    Fungsi untuk mendapatkan prediksi dari model
    :param X_test: <pandas dataframe> input
    :param y_test: <pandas series> output/target
    :param classifier: <sklearn method> model klasifikasi
    :param compute_score: <bool> True: menampilkan score, False: tidak
    :return test_predict: <list> hasil prediksi data input
    :return score: <float> akurasi model
    """
    if compute_score:
        score = classifier.score(X_test, y_test)
        print(f"Accuracy : {score:.4f}")

    test_predict = classifier.predict(X_test)

    return test_predict, score


# # Testing

# In[55]:


# Melakukan Testing

X_test_clean = extractTest(data = X_test,
                           numerical_column = numerical_column,
                           categorical_column = categorical_column,
                           ohe_column = ohe_columns,
                           standardizer = standardizer)


# In[56]:


X_test_clean.shape


# In[58]:


# Random Forest Performance
rf_test_predict, score = testPrediction(X_test = X_test_clean,
                                        y_test = y_test,
                                        classifier = random_forest,
                                        compute_score = True)


# In[59]:


# Random Forest 1 Performance
rf_1_test_predict, score = testPrediction(X_test = X_test_clean,
                                          y_test = y_test,
                                          classifier = random_forest_1,
                                          compute_score = True)  


# In[61]:


y_pred_proba = random_forest_1.predict_proba(X_test_clean)[:][:,1]

df_actual_predicted = pd.concat([pd.DataFrame(np.array(y_test), columns=['y_actual']), pd.DataFrame(y_pred_proba, columns=['y_pred_proba'])], axis=1)
df_actual_predicted.index = y_test.index


# In[62]:


# Melakukan Evaluasi performa dengan AUC

from sklearn.metrics import roc_curve, roc_auc_score

fpr, tpr, tr = roc_curve(df_actual_predicted['y_actual'], df_actual_predicted['y_pred_proba'])
auc = roc_auc_score(df_actual_predicted['y_actual'], df_actual_predicted['y_pred_proba'])

plt.plot(fpr, tpr, label='AUC = %0.4f' %auc)
plt.plot(fpr, fpr, linestyle = '--', color='k')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()


# In[63]:


df_actual_predicted = df_actual_predicted.sort_values('y_pred_proba')
df_actual_predicted = df_actual_predicted.reset_index()

df_actual_predicted['Cumulative N Population'] = df_actual_predicted.index + 1
df_actual_predicted['Cumulative N Bad'] = df_actual_predicted['y_actual'].cumsum()
df_actual_predicted['Cumulative N Good'] = df_actual_predicted['Cumulative N Population'] - df_actual_predicted['Cumulative N Bad']
df_actual_predicted['Cumulative Perc Population'] = df_actual_predicted['Cumulative N Population'] / df_actual_predicted.shape[0]
df_actual_predicted['Cumulative Perc Bad'] = df_actual_predicted['Cumulative N Bad'] / df_actual_predicted['y_actual'].sum()
df_actual_predicted['Cumulative Perc Good'] = df_actual_predicted['Cumulative N Good'] / (df_actual_predicted.shape[0] - df_actual_predicted['y_actual'].sum())


# In[64]:


# Melakukan Evaluasi performa dengan KS

KS = max(df_actual_predicted['Cumulative Perc Good'] - df_actual_predicted['Cumulative Perc Bad'])

plt.plot(df_actual_predicted['y_pred_proba'], df_actual_predicted['Cumulative Perc Bad'], color='r')
plt.plot(df_actual_predicted['y_pred_proba'], df_actual_predicted['Cumulative Perc Good'], color='b')
plt.xlabel('Estimated Probability for Being Bad')
plt.ylabel('Cumulative %')
plt.title('Kolmogorov-Smirnov:  %0.4f' %KS)


# In[ ]:





# In[ ]:





# In[ ]:




