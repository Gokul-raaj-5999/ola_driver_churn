import numpy as np
import pandas as pd
import streamlit as st
import re
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import Image
from six import StringIO
from sklearn.tree import export_graphviz
import pydot

from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from xgboost import XGBClassifier
import xgboost
from lightgbm import LGBMClassifier


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.metrics import confusion_matrix, classification_report, f1_score, roc_curve, roc_auc_score, precision_recall_curve, auc

st.set_page_config(
   page_title="OlaDriver",
   layout="wide",
   initial_sidebar_state="expanded",
)

data = pd.read_csv('ola_driver_scaler.csv')
df0 = data.copy()

st.title('Ola Driver Churn - CaseStudy')
st.write('Recruiting and retaining drivers is seen by industry watchers as a tough battle for Ola. Churn among drivers is high and it’s very easy for drivers to stop working for the service on the fly or jump to Uber depending on the rates. As the companies get bigger, the high churn could become a bigger problem. To find new drivers, Ola is casting a wide net, including people who don’t have cars for jobs. But this acquisition is really costly. Losing drivers frequently impacts the morale of the organization and acquiring new drivers is more expensive than retaining existing ones. You are working as a data scientist with the Analytics Department of Ola, focused on driver team attrition. You are provided with the monthly information for a segment of drivers for 2019 and 2020 and tasked to predict whether a driver will be leaving the company or not based on their attributes like Demographics (city, age, gender etc.) Tenure information (joining date, Last Date) Historical data regarding the performance of the driver (Quarterly rating, Monthly business acquired, grade, Income) Dataset:')
st.write('this is the dataSet given by Ola:',df0)

df0.drop(columns=['Unnamed: 0'], axis=1, inplace=True)

x = 100*df0.isnull().sum() / len(df0)
st.write('Null values:')
st.bar_chart(x)
st.write('\n- in this we have 91% null values in last working days so that is, the values are not present in the columns which means they are not leaving the company so , we can fill it with the 0\n- in age column also we have missing values that are filled with the preceding values, same for gender also. using ffill')

st.write('- in LastWorkingDate column we have a date of leaving that perticular date is not needed as we need only the value that is 1 or 0 so if we have that date in a row we fill that with 1 so that the driver is leaving that quater.\n- here we can split the reporting MMM-YY to r_day, r_month, r_year.\n- split the column date of joining to j_day, j_month, j_year.')

df0['LastWorkingDate'].fillna(value=0, inplace=True)
df0['LastWorkingDate'] = df0['LastWorkingDate'].apply(lambda x: 0 if x == 0 else 1)

df0['r_month'] = df0['MMM-YY'].apply(lambda x: int(str(x).split('/')[0]) )
df0['r_day'] = df0['MMM-YY'].apply(lambda x: int(str(x).split('/')[1]) )
df0['r_year'] = df0['MMM-YY'].apply(lambda x: int(str(x).split('/')[2]) )

df0['city'] = df0['City'].apply(lambda x: str(x)[1:] )

df0['j_month'] = df0['Dateofjoining'].apply(lambda x: int(str(x).split('/')[0]) )
df0['j_day'] = df0['Dateofjoining'].apply(lambda x: int(str(x).split('/')[1]) )
df0['j_year'] = df0['Dateofjoining'].apply(lambda x: int(str(x).split('/')[2]) )

df1 = df0[['r_day', 'r_month',
       'r_year','Driver_ID', 'Age', 'Gender', 'city', 'Education_Level',
       'Income', 'j_day', 'j_month', 'j_year', 'LastWorkingDate', 'Joining Designation',
       'Grade', 'Total Business Value', 'Quarterly Rating']]


imputer = KNNImputer(n_neighbors=2)
transformed = imputer.fit_transform(df1)
df2 = pd.DataFrame(transformed)

df2.rename(columns= {0:'r_day', 1:'r_month',2:'r_year',3:'Driver_ID',4:'Age',5:'Gender',6:'City',7:'Education_Level',8:'Income',
                     9:'j_day',10:'j_month',11:'j_year',12:'LastWorkingDate',13:'JoiningDesignation',14:'Grade',
                     15:'TotalBusinessValue',16:'QuarterlyRating'}, inplace=True)


df1['Age'].fillna(method= 'ffill', inplace=True)
df1['Gender'].fillna(method= 'ffill', inplace=True)
st.write('- as we can see that Knn imputer is not working goos as expected so we are using ffill for this method.')

# uni_aly = ['Age','Gender', 'city','Education_Level','Joining Designation', 'Grade', 'Quarterly Rating']
# count = 0
# plt.figure(figsize=(20,30))
# for i in uni_aly:
#     count += 1
#     plt.subplot(5,2,count)
#     sns.barplot( df1[i], )

st.subheader('Univarient Anaysis:')
t1, t2, t3, t4, t5, t6, t7 = st.tabs(['Age','Gender', 'city','Education_Level','Joining Designation', 'Grade', 'Quarterly Rating'])
with t1:
    st.bar_chart(df1, x='LastWorkingDate', y= 'Age')
with t2:
    st.bar_chart(df1, x='LastWorkingDate', y= 'Age')
with t3:
    st.bar_chart(df1, x='LastWorkingDate', y= 'city')
with t4:
    st.bar_chart(df1, x='LastWorkingDate', y= 'Education_Level')
with t5:
    st.bar_chart(df1, x='LastWorkingDate', y= 'Joining Designation')
with t6:
    st.bar_chart(df1, x='LastWorkingDate', y= 'Grade')
with t7:
    st.bar_chart(df1, x='LastWorkingDate', y= 'Quarterly Rating')

st.subheader('Outlier treatment:')
for col in ['Income', 'Total Business Value']:
  mean = df0[col].mean()
  std = df0[col].std()
  q1 = np.percentile(df0[col], 25)
  q2 = np.percentile(df0[col], 50)
  q3 = np.percentile(df0[col], 75)
  IQR = q3-q1
  lower_limt, upper_limit = q1-1.5*IQR , q3+1.5*IQR
  df0[col] = df0[col].apply(lambda x: lower_limt if x < lower_limt else x)
  df0[col] = df0[col].apply(lambda x: upper_limit if x > upper_limit else x)

# outliers = ['Income', 'Total Business Value']
# count = 0
# plt.figure(figsize=(20,30))
# for i in outliers:
#     count += 1
#     plt.subplot(5,3,count)
#     sns.boxplot(y= df0[i])

st.write('- in this outlies we will make the values that are more than upperwiskuss and liwerwiskuss to inside the range.\n- if we doo that what happens is that all the valus from total business value will compress and the mean value 0 will be shifted from 0 to higher value so that will also afect the output\n- even if we drop the null values then also we wont be left with more number of values. so in - this case its better to not treat.')

df2 = df0[['MMM-YY', 'Driver_ID', 'Age', 'Gender', 'City', 'Education_Level',
       'Income', 'Dateofjoining', 'LastWorkingDate', 'Joining Designation',
       'Grade', 'Total Business Value', 'Quarterly Rating']]

df2['City'] = df2['City'].apply(lambda x: int(str(x)[1:]) )
df2['MMM-YY'] = pd.to_datetime(df2['MMM-YY'])
df2['Dateofjoining'] = pd.to_datetime(df2['Dateofjoining'])
df2['TotalexpinDays'] = (df2['MMM-YY'] - df2['Dateofjoining']).dt.days

df3 = df2.groupby(by=['Driver_ID', "Gender", 'Dateofjoining', 'Joining Designation']).agg({'MMM-YY': 'count',
                                                                                          'Age' : 'max',
                                                                                          'City' : 'mean',
                                                                                          'Education_Level': 'max',
                                                                                          'Income': 'sum',
                                                                                          'LastWorkingDate': 'max',
                                                                                           'Grade': 'max',
                                                                                           'Total Business Value' : lambda x: list(x),
                                                                                           'Quarterly Rating': 'sum',
                                                                                           'TotalexpinDays':'max'}).reset_index()
df3.rename(columns={'MMM-YY' : 'TotalexpMonths', 'Income': 'tot_income'}, inplace=True)
df3['hasNegBusiValue'] = df3['Total Business Value'].apply(lambda x : 1 if min(x) < 0 else 0)
df3['totBusiValue'] = df3['Total Business Value'].apply(lambda x: sum(x))
df3['avg_income'] = df3['tot_income']/ df3['TotalexpMonths']
df4 = df3[['Gender', 'Joining Designation',
       'TotalexpMonths', 'Age', 'City', 'Education_Level', 'tot_income','avg_income',
        'Grade', 'Quarterly Rating',
       'TotalexpinDays', 'hasNegBusiValue', 'totBusiValue', 'LastWorkingDate']]

labelenc = LabelEncoder()

for i in ['Gender', 'Joining Designation', 'TotalexpMonths', 'Age', 'City','Education_Level', 'Grade','Quarterly Rating', 'hasNegBusiValue']:
  df4[i] = labelenc.fit_transform(df4[i])

X = df4.drop(columns=['LastWorkingDate'], axis=True)
y = df4['LastWorkingDate']

#as we have outliers we prefer StandardScaler over MinMaxScaler.
# scaler = StandardScaler()
# scaler = MinMaxScaler()
# X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
# X_train = scaler.fit_transform(X_train)

col1, col2 = st.columns(2)

with col1:
    st.subheader('ML Model : with Imbalance data')

    st.subheader('Logistic Regression')
    model = LogisticRegression()
    model.fit(X_train, y_train)
    st.write("training score",model.score(X_train, y_train))
    st.write("test score",model.score(X_test, y_test))
    y_pred = model.predict(X_test)
    st.write(pd.DataFrame(classification_report(y_test, y_pred,output_dict=True)))

    st.subheader('KNN classifier')
    model = KNeighborsClassifier()
    model.fit(X_train, y_train)
    st.write("training score",model.score(X_train, y_train))
    st.write("test score",model.score(X_test, y_test))
    y_pred = model.predict(X_test)
    st.write(pd.DataFrame(classification_report(y_test, y_pred,output_dict=True)))

    # y_train_score = []
    # y_test_score = []

    # for i in range(1,15):
    #   model = KNeighborsClassifier(n_neighbors=i)
    #   model.fit(X_train, y_train)
    #   y_train_score.append(model.score(X_train, y_train))
    #   y_test_score.append(model.score(X_test, y_test))

    # a = sns.lineplot(y_train_score, color='blue')
    # a = sns.lineplot(y_test_score, color='orange')
    # a = plt.axvline(x=y_test_score.index(max(y_test_score)), linestyle = '--', color= 'red')
    # fig = a.get_figure()
    # fig.savefig('img1.png')
    # st.image('img1.png')
    # st.write('maximum y test score', max(y_test_score))

    st.subheader('DecisionTree Classifier')
    features = list((df4.drop(columns=['LastWorkingDate'])).columns)
    model = DecisionTreeClassifier(criterion = 'gini')
    model.fit(X_train, y_train)
    st.write("training score",model.score(X_train, y_train))
    st.write("test score",model.score(X_test, y_test))
    y_pred = model.predict(X_test)
    st.write(pd.DataFrame(classification_report(y_test, y_pred,output_dict=True)))
    # fig = plt.figure(figsize=(20,10))
    # _ = tree.plot_tree(model, 
    #                    feature_names=features,
    #                    class_names=['0', '1'],
    #                    filled=True)

    # fig.savefig('img2.png')
    # st.image('img2.png')

    st.subheader('RandomForestClassifier')
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    st.write("training score",model.score(X_train, y_train))
    st.write("test score",model.score(X_test, y_test))
    y_pred = model.predict(X_test)
    st.write(pd.DataFrame(classification_report(y_test, y_pred,output_dict=True)))

with col2:
    st.subheader('ML Model : with Balanced data')
    smt = SMOTE()
    x_sm, y_sm = smt.fit_resample(X_train, y_train)

    st.subheader('Logistic Regression')
    model = LogisticRegression()
    model.fit(x_sm, y_sm)
    st.write("training score",model.score(x_sm, y_sm))
    st.write("test score",model.score(X_test, y_test))
    y_pred = model.predict(X_test)
    st.write(pd.DataFrame(classification_report(y_test, y_pred,output_dict=True)))

    st.subheader('KNN classifier')
    model = KNeighborsClassifier()
    model.fit(x_sm, y_sm)
    st.write("training score",model.score(x_sm, y_sm))
    st.write("test score",model.score(X_test, y_test))
    y_pred = model.predict(X_test)
    st.write(pd.DataFrame(classification_report(y_test, y_pred,output_dict=True)))

    st.subheader('DecisionTree Classifier')
    model = DecisionTreeClassifier()
    model.fit(x_sm, y_sm)
    st.write("training score",model.score(x_sm, y_sm))
    st.write("test score",model.score(X_test, y_test))
    y_pred = model.predict(X_test)
    st.write(pd.DataFrame(classification_report(y_test, y_pred,output_dict=True)))

    st.subheader('RandomForestClassifier')
    model = RandomForestClassifier()
    model.fit(x_sm, y_sm)
    st.write("training score",model.score(x_sm, y_sm))
    st.write("test score",model.score(X_test, y_test))
    y_pred = model.predict(X_test)
    st.write(pd.DataFrame(classification_report(y_test, y_pred,output_dict=True)))

st.write('conclussion \n 1. from this we come to know that balancing data is making to model to be better in slight manner but not as mush as more effective to train and use it.\n 1. when we compair with all the 4 classification algorithum, Random forest classification perform better in of all, as we can have both presession and recall to be btter for boht the labels, 0 and 1. \n\n results: before balancing to after balancing \n - logistic regression f1_score\n - knn classification f1_score\n - DecissionTree classification f1_score\n - RandomForest classification f1_score \n now will explecetly train the data for randum forest classification and train using its hyperparameters.')

#--------------------------------------------------------------------------------------

st.subheader('Hyperparameter Tunning :')
st.write('for hyperparameter tunning we use random forest with its hyperparameters.')
hyp_prams = {
    "n_estimators": [100,200,300,400,500],
    "max_depth" : [10, 20, 30,40,50,60,70,80,90,100]
}
rfc = RandomForestClassifier(criterion='gini', n_jobs=-1)
# model_hyp = GridSearchCV(rfc, hyp_prams)
model_hyp = RandomizedSearchCV(rfc, hyp_prams)
model_hyp.fit(X_train, y_train)
st.write(model_hyp.best_params_)

col1, col2 = st.columns(2)
with col1:
    st.write('without tuinning')
    model_rfc = RandomForestClassifier(criterion='gini', n_jobs=-1)
    model_rfc.fit(X_train, y_train)
    st.write("training score",model.score(X_train, y_train))
    st.write("test score",model.score(X_test, y_test))
    y_pred = model_rfc.predict(X_test)
    st.write(pd.DataFrame(confusion_matrix(y_test, y_pred)))

with col2:
    st.write('with tuinning')
    model_rfc = RandomForestClassifier(criterion='gini', n_jobs=-1, **model_hyp.best_params_)
    model_rfc.fit(X_train, y_train)
    st.write("training score",model.score(X_train, y_train))
    st.write("test score",model.score(X_test, y_test))
    y_pred = model_rfc.predict(X_test)
    st.write(pd.DataFrame(confusion_matrix(y_test, y_pred)))

#--------------------------------------------------------------------------------------
st.subheader('XGBoost :')
model_bost = XGBClassifier()
model_bost.fit(X_train, y_train)
st.write("training score",model.score(X_train, y_train))
st.write("test score",model.score(X_test, y_test))
y_pred = model_rfc.predict(X_test)
# st.write(pd.DataFrame(confusion_matrix(y_test, y_pred)))
# params = {
#         "n_estimators": [150,200, 250, 300],
#         "max_depth" : [2, 3, 4, 5, 7],
#         "learning_rate": [0.01, 0.02, 0.05, 0.07],
#         'subsample': [0.4, 0.5,0.6, 0.8],
#         'colsample_bytree': [0.6, 0.8, 1.0],
#         }
# xgb = XGBClassifier(objective='multi:softmax', num_class=20, silent=True)
# random_search = GridSearchCV( xgb, param_grid = params,scoring='accuracy',n_jobs=-1,cv=3)
# random_search.fit(X_train, y_train)
# st.write(random_search.best_params_)
# xgb = XGBClassifier(**random_search.best_params_ , num_classes=20)
# xgb.fit(X_train, y_train)
# st.write("training score",model.score(X_train, y_train))
# st.write("test score",model.score(X_test, y_test))
# y_pred = model_rfc.predict(X_test)
# st.write(pd.DataFrame(classification_report(y_test, y_pred,output_dict=True)))
# st.write(pd.DataFrame(confusion_matrix(y_test, y_pred)))

ax = sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap="crest", fmt='g')
ax.set(xlabel="Actual", ylabel="Preducted")
ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()
fig = ax.get_figure()
fig.savefig('img3.png')
st.image('img3.png')

st.write('Here we have predicted 305 events that are actually 1. We also predicted as 1. But actual 1 and we predicted 0 which has account of 64 so that is 1/6 of the total value. Which is even higher.')






























