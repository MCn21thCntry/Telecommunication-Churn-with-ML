####### ####### ####### ####### #######
####### Telco Churn Prediction #######
####### ####### ####### ####### #######

## --> İş Problemi:
# Şirketi terk edecek müşterileri tahmin edebilecek bir makine öğrenmesi modeli
# geliştirilmesi beklenmektedir.

## --> Veri Seti Hikayesi
# Telco müşteri kaybı verileri, üçüncü çeyrekte Kaliforniya'daki 7043 müşteriye ev telefonu ve İnternet hizmetleri sağlayan hayali
# bir telekom şirketi hakkında bilgi içerir. Hangi müşterilerin hizmetlerinden ayrıldığını, kaldığını veya hizmete kaydolduğunu
# gösterir.

## 21 Değişken 7043 Gözlem 977.5 KB
# CustomerId - Müşteri İd’si
# Gender - Cinsiyet
# SeniorCitizen - Müşterinin yaşlı olup olmadığı (1, 0)
# Partner - Müşterinin bir ortağı olup olmadığı (Evet, Hayır)
# Dependents - Müşterinin bakmakla yükümlü olduğu kişiler olup olmadığı (Evet, Hayır)
# tenure - Müşterinin şirkette kaldığı ay sayısı
# PhoneService - Müşterinin telefon hizmeti olup olmadığı (Evet, Hayır)
# MultipleLines - Müşterinin birden fazla hattı olup olmadığı (Evet, Hayır, Telefon hizmeti yok) --- Telefon hizmeti yok seçeneği silinebilir. Üsttekinden anlaşılır zaten
# InternetService - Müşterinin internet servis sağlayıcısı (DSL, Fiber optik, Hayır)
# OnlineSecurity - Müşterinin çevrimiçi güvenliğinin olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# OnlineBackup - Müşterinin online yedeğinin olup olmadığı (Evet, Hayır, İnternet hizmeti yok) -- burda da çıkarılabilir
# DeviceProtection - Müşterinin cihaz korumasına sahip olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# TechSupport - Müşterinin teknik destek alıp almadığı (Evet, Hayır, İnternet hizmeti yok) --- burda da çıkarılabilir
# StreamingTV - Müşterinin TV yayını olup olmadığı (Evet, Hayır, İnternet hizmeti yok) -- burda da çıkarılabilir
# StreamingMovies - Müşterinin film akışı olup olmadığı (Evet, Hayır, İnternet hizmeti yok) -- burda da çıkarılabilir
# Contract - Müşterinin sözleşme süresi (Aydan aya, Bir yıl, İki yıl)
# PaperlessBilling - Müşterinin kağıtsız faturası olup olmadığı (Evet, Hayır) -- Evet ise internet erişimi var demek olur fakat hayırların da erişimi olabilir
# PaymentMethod - Müşterinin ödeme yöntemi (Elektronik çek, Posta çeki, Banka havalesi (otomatik), Kredi kartı (otomatik))
# MonthlyCharges - Müşteriden aylık olarak tahsil edilen tutar
# TotalCharges - Müşteriden tahsil edilen toplam tutar
# Churn - Müşterinin kullanıp kullanmadığı (Evet veya Hayır)

import warnings
import numpy as np
import pandas as pd
import seaborn as sbn
from matplotlib import pyplot as plt
from pandas import get_dummies
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# !pip install catboost
# !pip install xgboost
# !pip install lightgbm

pd.set_option('display.max_columns', None)
pd.set_option("display.max_rows", None)
pd.set_option('display.width', 500)

warnings.simplefilter(action='ignore', category=Warning)

df = pd.read_csv("Telco-Customer-Churn.csv")

df["tenure"].sort_values(ascending=False).head(10)
df["tenure"].value_counts().sort_values(ascending=False).head(10)
df.loc[df["tenure"], "tenure"].value_counts().sort_values(ascending=False).head(10)

##################################
# GÖREV 1: KEŞİFCİ VERİ ANALİZİ
##################################

##################################
# GENEL RESİM
##################################

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
    for col in dataframe.columns:
        print(f"******* Is there any NaN(missing) value: {df.loc[df[col] == ' ', [col]].any()}")  #### New added
check_df(df)

### bağımlı değişkenimizi 1 ve 0 haline getirelim:
df["Churn"] = df["Churn"].apply(lambda x: 1 if x == "Yes" else 0)

df.loc[df["TotalCharges"]==" ", "TotalCharges"] ### 11 gözlemin toplam ücret kısmı boş #######4 Önemli 4########
"""# fakat "object" türünde kaydedildiği için isnull().sum()'da eksik değerinin olmadığı zannı uyandırıyor.
## float veya int türüne dönüştürdüğümüzde boş olan eksik değerler NaN olarak kaydedilmiş olarak gelecektir.
## İşte o zaman eksik değerlerimizin old. farketmiş oluruz."""

df.loc[df["TotalCharges"]==" ", :]
""" *** Bu 11'in tenure değerleri; 0, churn değerleri; "No" ve genç oldukları gözleniyor. 
*** Demekki, daha ay'ı bile dolmayan gözlemlerimiz var. *** """
## TotalCharges "object" tipinde, float değerler old. için float'a çevirmek gerekir.
#*** df["TotalCharges"] = df["TotalCharges"].astype("int64") ## hata veriyor sebebi aşağıda
### *** The code you provided attempts to convert the "TotalCharges" column in a DataFrame (df)
# to the int64 data type. However, this operation may fail if there are non-integer values or
# missing values (NaN) in the "TotalCharges" column. *** ###
""" The error you're encountering, "ValueError: invalid literal for int() with base 10: '29.85'",
 suggests that you're trying to convert a string containing a floating-point number ('29.85') 
 to an integer using the int() function. To resolve this issue, 
 you should convert the string to a float first and then to an integer if needed."""
##df["TotalCharges"] = df["TotalCharges"].astype(float) ### ValueError: could not convert string to float: ''

""" Üsttekiler olmadı sebebi; "TotalCharges" değeri boş olan yani eksik olan 11 gözlem. Tek çare aşağıdaki kod gibi!
 veya bu gözlemlerin doldurulup daha sonra df["TotalCharges"] = df["TotalCharges"].astype(float) uygulanması. """
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
df.info() ## float64 'e çevrilmiş durumda ve 11 değişkenimiz şuan eksik değer olarak gözükmekte.

df.isnull().sum() ### TotalCharges 11
df.loc[df["TotalCharges"].isnull(), "TotalCharges"] ########3 Önemli 3##########
tot_char_nan_indx = df.loc[df["TotalCharges"].isnull(), "TotalCharges"].index ######7 Önemli 7######

###### NaN olan 11 gözlemin değerini MonthlyCharges değerleriyle dolduralım:
df.loc[df["TotalCharges"].isnull(), "TotalCharges"] = df.loc[df["TotalCharges"].isnull(), "MonthlyCharges"] # Süper! Değerlerimizi atadık. #####6 Önemli 6#####


"""## Onun yerine tamamen silelim: Sonuçlar üsttekinden birazcık daha kötü o yüzden üsttekini tercih ettik.
df = df.drop(list(tot_char_nan_indx), axis=0) """

df.isnull().sum() ## TotalCharges 0
df.loc[tot_char_nan_indx, "TotalCharges"] ### Ve değerler güncellendi. ######8 Önemli 8######

df.loc[tot_char_nan_indx, :]
df["tenure"].value_counts()
df.loc[df["tenure"]==0, "tenure"] = 1 #### 2. yöntem
"""
df["tenure"] = df["tenure"].apply(lambda x: 1 if x==0 else x) ## ilk yöntem ####10 Önemli 10###
"""
########################################################################################################################
########################################################################################################################
########################################################################################################################
"""
df.loc[tot_char_nan_indx, :] ## 11 gözlem (All Contract: Two year)
# 488 --> DSL + OnlineSecurity + DeviceProtection + TechSupport + StreamingTV = 52.55
# 753 --> PhoneService = 20.25
# 936 --> PhoneService + DSL + OnlineSecurity + OnlineBackup + DeviceProtection + StreamingTV + StreamingMovies = 80.85
# 1082 --> PhoneService + MultipleLines = 25.75
# 1340 --> DSL + OnlineSecurity + OnlineBackup + DeviceProtection + TechSupport + StreamingTV = 56.05
# 3331 --> PhoneService = 19.85
# 3826 --> PhoneService + MultipleLines = 25.35
# 4380 --> PhoneService = 20.00
# 5218 --> PhoneService = 19.70 (Contract: 1 year)
# 6670 --> PhoneService + MultipleLines + DSL + OnlineBackup + DeviceProtection + TechSupport + StreamingTV = 73.35
# 6754 --> PhoneService + MultipleLines + DSL + OnlineSecurity + OnlineBackup + DeviceProtection + TechSupport = 61.90
df.loc[(df["MonthlyCharges"]==20.25) & (df["tenure"]==1), :] ## sadece PhoneService var. Yani Telefon Servis ücreti 20.25. (6 gözlem)
df.loc[(df["MonthlyCharges"]==80.85) & (df["tenure"]==1), :] ## PhoneService + Fiber optic + StreamingMovies = 80.85. (1 gözlem)
df.loc[(df["MonthlyCharges"]==25.75) & (df["tenure"]==1), :] ## PhoneService + MultipleLines = 25.75. (MultipleLines = 5.5)
df.loc[(df["MonthlyCharges"]==19.85) & (df["tenure"]==1), :] ## PhoneService = 19.85 (4 gözlem)
df.loc[(df["MonthlyCharges"]==25.35) & (df["tenure"]==1), :] ## DSL = 25.35 (1 gözlem)"""


##################################
# NUMERİK VE KATEGORİK DEĞİŞKENLERİN YAKALANMASI
##################################

def grab_col_names(dataframe, cat_th=10, car_th=20):

    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
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


##################################
# KATEGORİK DEĞİŞKENLERİN ANALİZİ
##################################

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sbn.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

for col in cat_cols:
    cat_summary(df, col)

### 1den daha fazla hattı olanların ortaklarının olup olmama oranı:
df[df["MultipleLines"]=="Yes"][["Partner"]].value_counts() #/ len(df[df["MultipleLines"]=="Yes"]) ## 2971
df.index # 7043
df[df["MultipleLines"]=="No"][["Partner"]].value_counts() #/ len(df[df["MultipleLines"]=="No"]) ## 3390
df.loc[df["MultipleLines"]=="No", ["Partner"]].value_counts()
df.loc[df["MultipleLines"]=="No phone service", "Partner"].value_counts()
### Demekki iş ortağı olanların tümü telefon kullanıyor.
""" İş ortağı olmayıp telefon kullanan oranlarına bakalım: """
df.loc[df["Partner"]=="No", ["PhoneService"]].value_counts() / len(df["Partner"]=="No")

df.loc[df["MultipleLines"] == "No phone service", ["PhoneService"]].value_counts()

df.loc[df["MultipleLines"]=="No phone service", :]

df["MultipleLines"].value_counts() # No phone service 682
df["PhoneService"].value_counts() # No 682
df.loc[df["MultipleLines"] == "No phone service", ["SeniorCitizen"]].value_counts()
df[["SeniorCitizen"]].value_counts()

"""### Bişey çıkmadı
df.loc[df["gender"]=="Male", ["Dependents"]].value_counts()
df.loc[df["gender"]=="Female", ["Dependents"]].value_counts()"""

##################################
# NUMERİK DEĞİŞKENLERİN ANALİZİ
##################################

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

for col in num_cols:
    num_summary(df, col, plot=True)
df.isnull().sum() ### Boş olan TotalCharges'lar farkedilmiyor! Yukarı da halletmiştik zaten.

# Contract(aya çevirelim) - tenure ilişkisini inceleyelim
df.loc[:, ["Contract","tenure"]].head(15)
df.loc[:, ["TotalCharges","tenure", "Churn"]].sort_values(by="TotalCharges", ascending=False).head(100)  ####5 Önemli 5####
df["Churn"].value_counts()

df["Contract"].value_counts()
df.groupby("Contract").agg({"MonthlyCharges": ["count", "mean"],
                            "TotalCharges": ["count", "mean"]})

df.groupby("PaymentMethod").agg({"MonthlyCharges": ["count", "mean"],
                            "TotalCharges": ["count", "mean"]})

df.loc[:, ["OnlineSecurity","DeviceProtection","tenure","MonthlyCharges", "TotalCharges"]].head(15)
df.head(15)

# PaperlessBilling - Müşterinin kağıtsız faturası olup olmadığı (Evet, Hayır) -- Evet ise internet erişimi var demek olur fakat hayırların da erişimi olabilir
# PaperlessBilling - InternetService ---> kıyaslayalım
df.groupby("PaperlessBilling")["InternetService","PhoneService"].value_counts() ####2 Önemli 2####
## Sonuç alamadık.

### PaymentMethod --- PaperlessBilling ---> hangi methodun online old. ve internet erişimi gerektirdiğini gösterir.
# Eğer InternetService yok ve ödeme online ise bu "telefon internetinin" old. gösterir.
df.groupby("PaperlessBilling")["InternetService","PaymentMethod"].value_counts()


# Tenure'e bakıldığında 1 aylık müşterilerin çok fazla olduğunu ardından da 70 aylık müşterilerin geldiğini görüyoruz.
# Farklı kontratlardan dolayı gerçekleşmiş olabilir, aylık sözleşmesi olan kişilerin tenure ile 2 yıllık sözleşmesi olan kişilerin tenure'ne bakalım.
df[df["Contract"] == "Month-to-month"]["tenure"].hist(bins=20)
plt.xlabel("tenure")
plt.title("Month-to-month")
plt.show()

df[df["Contract"] == "Two year"]["tenure"].hist(bins=20)
plt.xlabel("tenure")
plt.title("Two year")
plt.show()

# MonthyChargers'a bakıldığında aylık sözleşmesi olan müşterilerin aylık ortalama ödemeleri daha fazla olabilir.
df[df["Contract"] == "Month-to-month"]["MonthlyCharges"].hist(bins=20)
plt.xlabel("MonthlyCharges")
plt.title("Month-to-month")
plt.show()

df[df["Contract"] == "Two year"]["MonthlyCharges"].hist(bins=20)
plt.xlabel("MonthlyCharges")
plt.title("Two year")
plt.show()



##################################
# NUMERİK DEĞİŞKENLERİN TARGET GÖRE ANALİZİ
##################################

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: ["mean", "count"]}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df, "Churn", col)


##################################
# KATEGORİK DEĞİŞKENLERİN TARGET GÖRE ANALİZİ
##################################

def target_summary_with_cat(dataframe, target, categorical_col):
    print(categorical_col)
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean(),
                        "Count": dataframe[categorical_col].value_counts(),
                        "Ratio": 100 * dataframe[categorical_col].value_counts() / len(dataframe)}), end="\n\n\n")

for col in cat_cols:
    target_summary_with_cat(df, "Churn", col)

df.loc[df["Contract"]=="Two year", ["Churn"]].value_counts()
df.loc[df["Contract"]=="One year", ["Churn"]].value_counts()
df.loc[df["Contract"]=="Month-to-month", ["Churn"]].value_counts()
####### Sınıflarını numerik yapalım:
df["Contract"] = df["Contract"].apply(lambda x: 1 if x=="Month-to-month" else 12 if x=="One year" else 24) #######9 Önemli 9########
df["Contract"].value_counts()
df.loc[df["Contract"]==24, ["Churn"]].value_counts()
df.loc[df["Contract"]==12, ["Churn"]].value_counts()
df.loc[df["Contract"]==1, ["Churn"]].value_counts()

"""### Yeni değişken oluşturalım: --- Değişken oluşturma kısmına taşındı.
df["NEW_TOTAL_CONTRACT_CHARGE"] = df["MonthlyCharges"] * df["Contract"] * 12"""


# Adım 1: Numerik ve kategorik değişkenleri yakalayınız.
# Adım 2: Gerekli düzenlemeleri yapınız. (Tip hatası olan değişkenler gibi)
# Adım 3: Numerik ve kategorik değişkenlerin veri içindeki dağılımını gözlemleyiniz.
# Adım 4: Kategorik değişkenler ile hedef değişken incelemesini yapınız.
# Adım 5: Aykırı gözlem var mı inceleyiniz.
# Adım 6: Eksik gözlem var mı inceleyiniz

################################################
# 2. Data Preprocessing & Feature Engineering
################################################

# Adım 1: Eksik ve aykırı gözlemler için gerekli işlemleri yapınız.


#################################################
# Adım 2: Yeni değişkenler oluşturunuz.
#################################################

df.groupby(["InternetService"]).agg({"MonthlyCharges": ["count", "mean"],
                            "TotalCharges": ["count", "mean"]})
######### Internet servisi olmayanlarım aylık ve toplam ödeme miktarları diğerlerinin oldukça altında
df.groupby(["PhoneService"]).agg({"MonthlyCharges": ["count", "mean"],
                            "TotalCharges": ["count", "mean"]})

df.loc[df["InternetService"]=="DSL"]["PhoneService"].value_counts() ####1 Önemli 1####
#### Telefonu olmayanların tümü DSL interneti tercih ediyor. Ayrıca, Fiber DSL'den daha pahalıydı.
# Bu, telefonu olmayanların maddi durumlarının iyi olmadığı anlamına geliyor.
""" maddi durumu iyi veya değil değişkeni oluşturulabilir. """ ## Fakat iki aşağıda interneti olmayanların
# daha muhtaç durumda oldukları göz önüne alındığında;
""" *************** İlk yeni değişken ve sınıflarımız: ******************** """
df.loc[df["InternetService"]=="No", "NEW_Welfare"] = "Worse"
df.loc[df["PhoneService"]=="No", "NEW_Welfare"] = "Bad"
df.loc[(df["InternetService"]=="DSL") & (df["PhoneService"]=="Yes"), "NEW_Welfare"] = "Good"
df.loc[df["InternetService"]=="Fiber", "NEW_Welfare"] = "Better"

### hem internet hem de telefon servisi olmayanlar yok. Yani ya her ikisi ya da biri olup diğerinin olmaması durumu var.
########## Internet servisi olmayanları -- Telefon servisi olmayanları karşılaştıralım:
df.groupby(df["InternetService"]=="No").agg({"MonthlyCharges": ["count", "mean"],
                            "TotalCharges": ["count", "mean"]})
df.groupby(df["PhoneService"]=="No").agg({"MonthlyCharges": ["count", "mean"],
                            "TotalCharges": ["count", "mean"]})
### İnterneti olmayanların aylık ve toplam ücret ödemeleri telefonları olmasına rağmen çok çok düşük.
df["PhoneService"].value_counts() ## No 682 ---> 7043 'ten
df["InternetService"].value_counts() ## No  1526 ---> 7043 'ten
""" ****************** İkinci değişken ve sınıflarımız: ******************* """
# PhoneService - Müşterinin telefon hizmeti olup olmadığı (Evet, Hayır)
# MultipleLines - Müşterinin birden fazla hattı olup olmadığı (Evet, Hayır, Telefon hizmeti yok) --- Telefon hizmeti yok seçeneği silinebilir. Üsttekinden anlaşılır zaten
df["MultipleLines"].value_counts()
df.loc[df["MultipleLines"]=="No phone service", "NEW_Numberof_Phone"] = 0
df.loc[df["MultipleLines"]=="No", "NEW_Numberof_Phone"] = 1
df.loc[df["MultipleLines"]=="Yes", "NEW_Numberof_Phone"] = 2
### df["MultipleLines"] = LabelEncoder().fit_transform(df["MultipleLines"]) ## yanlış değerler atıyor. """
df["NEW_Numberof_Phone"].value_counts()

""" ****************** Üçüncü değişken ve sınıflarımız: ******************* """
df.loc[df["SeniorCitizen"]==0, ["Dependents"]].value_counts()
df.loc[df["SeniorCitizen"]==1, ["Dependents"]].value_counts()
df.loc[(df["SeniorCitizen"]==1) & (df["Dependents"]=="Yes"), "NEW_IS_ALONE"] = "SENIOR_WITH_FAMILY"
df.loc[(df["SeniorCitizen"]==1) & (df["Dependents"]=="No"), "NEW_IS_ALONE"] = "SENIOR_NO_FAMILY"
df.loc[(df["SeniorCitizen"]==0) & (df["Dependents"]=="Yes"), "NEW_IS_ALONE"] = "YOUNG_WITH_FAMILY"
df.loc[(df["SeniorCitizen"]==0) & (df["Dependents"]=="No"), "NEW_IS_ALONE"] = "YOUNG_NO_FAMILY"

""" ****************** Dördüncü, Beşinci ve Altıncı değişkenlerimiz: ******************* """
df["NEW_AVERAGE_MONTHLY_CHARGE"] = df["TotalCharges"] / df["tenure"] ## average monthly payment so far
df["NEW_CHARGE_RISE"] = df["MonthlyCharges"] - df["NEW_AVERAGE_MONTHLY_CHARGE"]  ## approximate monthly charge rise (göze alınan ödeme farkı miktar)
df["NEW_RISE_TOT_IMPACT"] = df["NEW_CHARGE_RISE"] / df["TotalCharges"] ## TotalCharges'ın ödenecek yeni tutar farkına etkisi

""" ****************** Yedinci değişkenimiz: ******************* """
df["NEW_TOTAL_CONTRACT_CHARGE"] = df["MonthlyCharges"] * df["Contract"] ## Göze alınan ödeme tutarı

cat_cols, num_cols, cat_but_car = grab_col_names(df)
######################################
# Adım 3: Encoding işlemlerini gerçekleştiriniz.
######################################

# InternetService - Müşterinin internet servis sağlayıcısı (DSL, Fiber optik, Hayır)
# OnlineSecurity - Müşterinin çevrimiçi güvenliğinin olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
df.groupby(df["OnlineSecurity"]=="Yes")["InternetService"].value_counts() ####2 Önemli 2####
df.loc[:, ["OnlineSecurity","OnlineBackup","InternetService","DeviceProtection", "TechSupport"]].head(15)
#### İnternet erişimi yoksa üstteki hizmetlerde yok!

# OnlineBackup - Müşterinin online yedeğinin olup olmadığı (Evet, Hayır, İnternet hizmeti yok) -- burda da çıkarılabilir
# TechSupport - Müşterinin teknik destek alıp almadığı (Evet, Hayır, İnternet hizmeti yok) --- burda da çıkarılabilir

cat_cols, num_cols, cat_but_car = grab_col_names(df)
df.head()
####### Sınıf sayısını düşürme ve Encoding
online_help_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "PhoneService"]

df["OnlineSecurity"].value_counts()
df[df["OnlineSecurity"]=="No"]["OnlineSecurity"].value_counts()
for col in online_help_cols:
    df.loc[df[col]=="No internet service", col]="No" ## sadece col'u değiştir. col yazılmazsa tüm değişkenleri etkiler.
    df[col] = LabelEncoder().fit_transform(df[col])
df["OnlineSecurity"].value_counts()
## Yeni değişken türetme
df["NEW_ONLINE_HELP"] = df["OnlineSecurity"] + df["OnlineBackup"] + df["DeviceProtection"] + df["TechSupport"] + df["PhoneService"] #+ df["InternetService"]

cat_cols, num_cols, cat_but_car = grab_col_names(df)
df.head()
df.shape
binary_cols = [col for col in cat_cols if (df[col].dtype == "O") and (df[col].nunique() == 2)]
df.head()
for col in binary_cols:
    lab_enc = LabelEncoder()
    df[col] = lab_enc.fit_transform(df[col])

len(binary_cols)
cat_cols, num_cols, cat_but_car = grab_col_names(df)


# Tenure  değişkeninden yıllık kategorik değişken oluşturma
df.loc[(df["tenure"]>=0) & (df["tenure"]<=12),"NEW_TENURE_YEAR"] = "0-1 Year"
df.loc[(df["tenure"]>12) & (df["tenure"]<=24),"NEW_TENURE_YEAR"] = "1-2 Year"
df.loc[(df["tenure"]>24) & (df["tenure"]<=36),"NEW_TENURE_YEAR"] = "2-3 Year"
df.loc[(df["tenure"]>36) & (df["tenure"]<=48),"NEW_TENURE_YEAR"] = "3-4 Year"
df.loc[(df["tenure"]>48) & (df["tenure"]<=60),"NEW_TENURE_YEAR"] = "4-5 Year"
df.loc[(df["tenure"]>60) & (df["tenure"]<=72),"NEW_TENURE_YEAR"] = "5-6 Year"

df.groupby("NEW_TENURE_YEAR")["NEW_AVG_Charges"].mean()

# Kontratı 1 veya 2 yıllık müşterileri Engaged olarak belirtme
df["NEW_Engaged"] = df["Contract"].apply(lambda x: 1 if x in [12, 24] else 0)

# Herhangi bir destek, yedek veya koruma almayan kişiler
df["NEW_noProt"] = df.apply(lambda x: 1 if (x["OnlineBackup"] != 1) or (x["DeviceProtection"] != 1) or (x["TechSupport"] != 1) or (x["OnlineSecurity"] != 1) else 0, axis=1)

# Aylık sözleşmesi bulunan ve genç olan müşteriler
df["NEW_Young_Not_Engaged"] = df.apply(lambda x: 1 if (x["NEW_Engaged"] == 0) and (x["SeniorCitizen"] == 0) else 0, axis=1)

df.head()
# Kişinin toplam aldığı servis sayısı
df['NEW_TotalServices'] = (df[['PhoneService', 'InternetService', 'OnlineSecurity',
                                       'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                       'StreamingTV', 'StreamingMovies']]== 1).sum(axis=1)

# Herhangi bir streaming hizmeti alan kişiler
df["NEW_FLAG_ANY_STREAMING"] = df.apply(lambda x: 1 if (x["StreamingTV"] == 1) or (x["StreamingMovies"] == 1) else 0, axis=1)

# Kişi otomatik ödeme yapıyor mu?
df["NEW_FLAG_AutoPayment"] = df["PaymentMethod"].apply(lambda x: 1 if x in ["Bank transfer (automatic)","Credit card (automatic)"] else 0)

# ortalama aylık ödeme
df["NEW_AVG_Charges"] = df["TotalCharges"] / (df["tenure"])

df.loc[df["Contract"]==1,"NEW_AVG_Charges"].sort_values(ascending=True).tail(10)
df.loc[df["Contract"]==12,"NEW_AVG_Charges"].sort_values(ascending=True).tail(10)
df.loc[df["Contract"]==24,"NEW_AVG_Charges"].sort_values(ascending=True).tail(10)
df["NEW_AVG_Charges"].sort_values(ascending=True)

# Servis başına ücret
df["NEW_AVG_Service_Fee"] = df["MonthlyCharges"] / (df['NEW_TotalServices']+1) ## 1 eklenmese inf değerler üretilir. Aşağıda icabına bakılabilecek birkaç şeyler var.
df.corrwith(df["Churn"]).sort_values(ascending=False)

df.info()
cat_cols, num_cols, cat_but_car = grab_col_names(df)
remain_cols = [col for col in cat_cols if len(df[col].unique())>2 and df[col].dtypes in ["object"]]
""" ****** One Hot Encoding ****** """
df = pd.get_dummies(df, columns=remain_cols, drop_first=True)
cat_cols, num_cols, cat_but_car = grab_col_names(df)

df["TotalCharges"].sort_values(ascending=False)
df.loc[df["Contract"]==1,"TotalCharges"].sort_values(ascending=False)

df.groupby("Contract")["TotalCharges"].mean()
df.groupby("Contract")["NEW_AVG_Charges"].mean()

df.groupby("NEW_TENURE_YEAR")["NEW_AVG_Charges"].mean()

df.loc[df["NEW_TENURE_YEAR"]=="4-5 Year", ["Contract","NEW_AVG_Charges"]].value_counts().head(12)

df.loc[df["NEW_TENURE_YEAR"]=="5-6 Year", "Contract"].value_counts().head(12)

len(df[df["NEW_TENURE_YEAR"]=="5-6 Year"])

"""### Sözleşmesiz ödeyenlerin yıllara göre oranlarıyla yeni değişken: """
df.loc[:,["NEW_TENURE_YEAR", ["NEW_UNCONTRACT_RATE_PER_YEAR"]]] = df.loc[df["Contract"]==1,"NEW_TENURE_YEAR"].value_counts() / df["NEW_TENURE_YEAR"].value_counts()
df["NEW_TENURE_YEAR"].value_counts()

df.loc[(df["Contract"]==1) & (df["NEW_TENURE_YEAR"]=="0-1 Year"), "NEW_UNCONTRACT_RATE_PER_YEAR"] = df.loc[(df["Contract"]==1) & (df["NEW_TENURE_YEAR"]=="0-1 Year")] / len(df[df["NEW_TENURE_YEAR"]=="0-1 Year"])
df["NEW_UNCONTRACT_RATE_PER_YEAR"] = df.loc[df["Contract"]==12,"NEW_TENURE_YEAR"].value_counts() / df["NEW_TENURE_YEAR"].value_counts()
df["NEW_UNCONTRACT_RATE_PER_YEAR"] = df.loc[df["Contract"]==24,"NEW_TENURE_YEAR"].value_counts() / df["NEW_TENURE_YEAR"].value_counts()
df.loc[:,["NEW_UNCONTRACT_RATE_PER_YEAR", "Contract"]].head()
df["NEW_UNCONTRACT_RATE_PER_YEAR"].isnull().sum()
len(df.loc[df["Contract"]==1,"NEW_TENURE_YEAR"])


""" Sözleşmesi olmadan aylık ödeyen kişilerin aylık ödeme miktarlarının daha fazla olduğu gözlemiyle;
Ortalama aylık ödeme miktarıyla şimdiye kadar ne derece sözleşmeli ödeyip ödemediği ihtimallerini bir dereceye kadar ortaya çıkaralım: """
len(df.loc[(df["Contract"]==1) & (df["NEW_AVG_Charges"])]) # 3875
df.loc[(df["Contract"]==1) & (df["NEW_AVG_Charges"]),"NEW_HIST_PAYMENT_METH"] = df[df["Contract"]==1]["NEW_AVG_Charges"].mean()
df.loc[(df["Contract"]==12) & (df["NEW_AVG_Charges"]),"NEW_HIST_PAYMENT_METH"] = df[df["Contract"]==12]["NEW_AVG_Charges"].mean()
df.loc[(df["Contract"]==24) & (df["NEW_AVG_Charges"]),"NEW_HIST_PAYMENT_METH"] = df[df["Contract"]==24]["NEW_AVG_Charges"].mean()
len(df.loc[df["Contract"]==1,"NEW_AVG_Charges"]) # 3875
len(df.loc[df["Contract"]==12,"NEW_AVG_Charges"]) # 1473
len(df.loc[df["Contract"]==24,"NEW_AVG_Charges"]) # 1695
df["Contract"].value_counts()

df.corrwith(df["Churn"]).sort_values(ascending=False)

df.loc[:15, ["NEW_HIST_PAYMENT_METH","Contract"]]
df.head()
df.isnull().sum()
""" ************************ ************************ Çok Önemli! ******************** ****************"""
## Select only numeric columns
numeric_columns_df = df.select_dtypes(include=[np.number]) ########11 Önemli 11##########
type(numeric_columns_df) ## pandas.core.frame.DataFrame
##### checking inf in numeric columns in df because cannot check in any categoric col
inf_check = np.isinf(numeric_columns_df)
inf_check.any()  ### 1 eklemediğimizde NEW_AVG_Service_Fee   True'du. Şuan False
inf_check.any(axis=1) ### tüm gözlemlerin inf içerip içermeme durumu #####12 Önemli 12#####
inf_check.any().any()  ### False
""""
# Replace infinite values with NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)
inf_check.any().any()  ### True
df.isnull().sum()
************************ ************************ ************************* ******************** ****************"""

"""df.groupby("PaperlessBilling").agg({"tenure":[ "mean", "count"],
                                    "Churn": ["mean", "count"]})"""

################################################
# Adım 4: Numerik değişkenler için standartlaştırma yapınız.
################################################
cat_cols, num_cols, cat_but_car = grab_col_names(df)
df.head()
""" There is a problem with scaling
for col in num_cols:
    scale = StandardScaler()
    df[col] = scale.fit_transform(df[col])"""

df.corrwith(df["Churn"]).sort_values(ascending=False)

################################################
# 3. Modeling
################################################

y = df["Churn"]
X = df.drop(["Churn","customerID"], axis=1)


models = [('LR', LogisticRegression(random_state=12345)),
          ('KNN', KNeighborsClassifier()),
          ('CART', DecisionTreeClassifier(random_state=12345)),
          ('RF', RandomForestClassifier(random_state=12345)),
          ('SVM', SVC(gamma='auto', random_state=12345)),
          ('XGB', XGBClassifier(random_state=12345)),
          ("LightGBM", LGBMClassifier(random_state=12345)),
          ("CatBoost", CatBoostClassifier(verbose=False, random_state=12345))]

for name, model in models:
    cv_results = cross_validate(model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])
    print(f"########## {name} ##########")
    print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)}")
    print(f"Auc: {round(cv_results['test_roc_auc'].mean(), 4)}")
    print(f"Recall: {round(cv_results['test_recall'].mean(), 4)}")
    print(f"Precision: {round(cv_results['test_precision'].mean(), 4)}")
    print(f"F1: {round(cv_results['test_f1'].mean(), 4)}")

""" 
########## LR ##########
Accuracy: 0.8055
Auc: 0.8435
Recall: 0.5356
Precision: 0.6664
F1: 0.5935
########## KNN ##########
Accuracy: 0.7758
Auc: 0.774
Recall: 0.4698
Precision: 0.5997
F1: 0.5265
########## CART ##########
Accuracy: 0.7255
Auc: 0.6503
Recall: 0.4858
Precision: 0.4835
F1: 0.4839
########## RF ##########
Accuracy: 0.7921
Auc: 0.8277
Recall: 0.4949
Precision: 0.6424
F1: 0.5585
########## SVM ##########
Accuracy: 0.766
Auc: 0.6739
Recall: 0.2135
Precision: 0.696
F1: 0.3257
########## XGB ##########
Accuracy: 0.7843
Auc: 0.8238
Recall: 0.4901
Precision: 0.6188
F1: 0.5467
########## LightGBM ##########
Accuracy: 0.7944
Auc: 0.837
Recall: 0.5195
Precision: 0.64
F1: 0.573
########## CatBoost ##########
Accuracy: 0.7985
Auc: 0.8406
Recall: 0.5137
Precision: 0.654
F1: 0.5748
"""

""" Sadece Benim yazdığım yeni değişkenler ile
########## LR ##########
Accuracy: 0.8063
Auc: 0.8444
Recall: 0.5511
Precision: 0.6626
F1: 0.6013
########## KNN ##########
Accuracy: 0.7735
Auc: 0.7664
Recall: 0.481
Precision: 0.5911
F1: 0.53
########## CART ##########
Accuracy: 0.7268
Auc: 0.6563
Recall: 0.5013
Precision: 0.4858
F1: 0.4933
########## RF ##########
Accuracy: 0.794
Auc: 0.8305
Recall: 0.4981
Precision: 0.6454
F1: 0.5619
########## SVM ##########
Accuracy: 0.7673
Auc: 0.6687
Recall: 0.2392
Precision: 0.6759
F1: 0.3525
########## XGB ##########
Accuracy: 0.783
Auc: 0.8208
Recall: 0.503
Precision: 0.6121
F1: 0.5519
########## LightGBM ##########
Accuracy: 0.7953
Auc: 0.8355
Recall: 0.5265
Precision: 0.6396
F1: 0.5769
########## CatBoost ##########
Accuracy: 0.8011
Auc: 0.8413
Recall: 0.519
Precision: 0.6604
F1: 0.5807
"""



################################################
# Random Forests
################################################

rf_model = RandomForestClassifier(random_state=17)

rf_params = {"max_depth": [5, 8, None],
             "max_features": [3, 5, 7, "auto"],
             "min_samples_split": [2, 5, 8, 15, 20],
             "n_estimators": [100, 200, 500]}

rf_best_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

rf_best_grid.best_params_

rf_best_grid.best_score_

rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state=17).fit(X, y)


cv_results = cross_validate(rf_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()


################################################
# XGBoost
################################################

xgboost_model = XGBClassifier(random_state=17)

xgboost_params = {"learning_rate": [0.1, 0.01, 0.001],
                  "max_depth": [5, 8, 12, 15, 20],
                  "n_estimators": [100, 500, 1000],
                  "colsample_bytree": [0.5, 0.7, 1]}

xgboost_best_grid = GridSearchCV(xgboost_model, xgboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

xgboost_final = xgboost_model.set_params(**xgboost_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(xgboost_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()


################################################
# LightGBM
################################################

lgbm_model = LGBMClassifier(random_state=17)

lgbm_params = {"learning_rate": [0.01, 0.1, 0.001],
               "n_estimators": [100, 300, 500, 1000],
               "colsample_bytree": [0.5, 0.7, 1]}

lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(lgbm_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()



################################################
# CatBoost
################################################

catboost_model = CatBoostClassifier(random_state=17, verbose=False)

catboost_params = {"iterations": [200, 500],
                   "learning_rate": [0.01, 0.1],
                   "depth": [3, 6]}

catboost_best_grid = GridSearchCV(catboost_model, catboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

catboost_final = catboost_model.set_params(**catboost_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(catboost_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()


################################################
# Feature Importance
################################################

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sbn.set(font_scale=1)
    sbn.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(rf_final, X)
plot_importance(xgboost_final, X)
plot_importance(lgbm_final, X)
plot_importance(catboost_final, X)






