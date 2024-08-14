import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
plt.style.use('ggplot')
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
data = pd.read_csv('web-page-phishing.csv')
data.head()
data.describe()
descriptive_stats = data.iloc[:,[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,19]].groupby('phishing').describe()
stat_data = descriptive_stats.xs('mean', level=1, axis=1)
stat_data = np.sqrt(stat_data)
n_features = len(stat_data.columns)
index = np.arange(n_features)
bar_width = 0.35
plt.figure(figsize=(14, 8))
plt.bar(index, stat_data.iloc[0], bar_width, label='Non-Phishing')
plt.bar(index + bar_width, stat_data.iloc[1], bar_width, label='Phishing')
plt.xlabel('Features')
plt.ylabel('Mean Value')
plt.title('Mean Values of Features by Phishing Category')
plt.xticks(index + bar_width / 2, stat_data.columns, rotation=90)
plt.legend()
plt.tight_layout()
plt.show()
plt.figure(figsize=(20,10))
sns.heatmap(data.corr(),cmap='coolwarm',annot=True,cbar=False)
plt.show()
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
X = data.drop(['phishing'],axis=1).values
y = data['phishing'].values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1)
model=RandomForestClassifier(n_estimators=80,max_depth=18,max_features='sqrt',min_samples_split=12,criterion='gini')
model.fit(X_train,y_train)
predictions_train = model.predict(X_train)
predictions_test = model.predict(X_test)
print(f"Train Accuracy Score: {accuracy_score(y_train,predictions_train)}")
print(f"Test Accuracy Score: {accuracy_score(y_test,predictions_test)}")
def confusion_matrix_plot(y_test,predictions):
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", 
                cmap="Blues",
                square=True,
                cbar=False,
                xticklabels=['False', 'True'],
                yticklabels=['False', 'True'])
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(2) + 0.5
    plt.xticks(tick_marks, ['Non-phishing', 'Phishing'], rotation=0)
    plt.yticks(tick_marks, ['Non-phishing', 'Phishing'], rotation=0)
    plt.tight_layout()
    plt.show()
def calculate_metrics(model,X_test,y_test):
    predictions = model.predict(X_test)
    print(f"Accuracy Score: {accuracy_score(y_test,predictions)}")
    confusion_matrix_plot(y_test,predictions)
  calculate_metrics(model,X_test,y_test)
