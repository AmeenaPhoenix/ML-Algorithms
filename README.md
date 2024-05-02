# ML-Algorithms
The dataset is a classic example for learning KNN classification, featuring demographic attributes like age, education, gender, and occupation. It's valuable for practicing data preprocessing and understanding the impact of socio-economic factors on income levels. The goal is to predict an individual's annual income based on these factors.

# Importing the libraries 

import pandas as pd
import numpy as np 
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# Ignore harmless warnings 

import warnings 
warnings.filterwarnings("ignore")

# Set to display all the columns in dataset

pd.set_option("display.max_columns", None)

# Import psql to run queries 

#import pandasql as psql

AOdata.duplicated().any()

AOdata.isnull().sum()

AOdata.nunique()

AOdata = AOdata.drop(['x', 'fnlwgt', 'educational-num', 'capital-gain'], axis=1)
AOdata.info()

AOdata["workclass"].value_counts()

AOdata["occupation"].value_counts()

AOdata["native-country"].value_counts()

AOdata['workclass'].value_counts()

AOdata.isnull().sum()


# Use LabelEncoder to handle categorical data

from sklearn.preprocessing import LabelEncoder

LE = LabelEncoder()

for col in AOdata.columns:
    if AOdata[col].dtypes=='object':
        AOdata[col]=LE.fit_transform(AOdata[col])
AOdata.head()

AOdata['income'].value_counts()


# Count the target or dependent variable by '0' & '1' and their proportion  
# (> 10 : 1, then the dataset is imbalance data) 
      
AOdata_count = AOdata.income.value_counts() 
print('Class 0:', AOdata_count[0]) 
print('Class 1:', AOdata_count[1]) 
print('Proportion:', round(AOdata_count[0] /AOdata_count[1], 2), ': 1') 
print('Total counts in income:', len(AOdata))


# Identifincome the Independent and Target variables

IndepVar = []
for col in AOdata.columns:
    if col != 'income':
        IndepVar.append(col)

TargetVar = 'income'

x = AOdata[IndepVar]
y = AOdata[TargetVar]


# Splitting the dataset into train and test 

from sklearn.model_selection import train_test_split

#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.30, random_state = 42)

x_train,x_test,y_train,y_test = train_test_split(x, y,test_size=0.2,stratify=y,random_state=42)

# Display the shape of train and test data 

x_train.shape, x_test.shape, y_train.shape, y_test.shape


# Scaling the features by using MinMaxScaler

from sklearn.preprocessing import MinMaxScaler

mmscaler = MinMaxScaler(feature_range=(0, 1))

x_train = mmscaler.fit_transform(x_train)
x_train = pd.DataFrame(x_train)

x_test = mmscaler.fit_transform(x_test)
x_test = pd.DataFrame(x_test)


x_train.head()


#KNN ALGORITHM
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, roc_auc_score
import numpy as np

accuracy = []
new_row = []

for k in range(1, 21):
    # Build the model
    ModelKNN = KNeighborsClassifier(n_neighbors=k)
    
    # Train the model
    ModelKNN.fit(x_train, y_train)
    
    # Convert x_test to numpy array if it's not already
    x_test_np = np.array(x_test)
    
    # Predict the model
    y_pred = ModelKNN.predict(x_test_np)
    y_pred_prob = ModelKNN.predict_proba(x_test_np)
    
    print('KNN_K_value = ', k)
    print('Model Name: ', ModelKNN)
    
    # Confusion matrix
    matrix = confusion_matrix(y_test, y_pred, labels=[1,0])
    print('Confusion matrix:\n', matrix)
    
    # Classification report
    c_report = classification_report(y_test, y_pred, labels=[1,0])
    print('Classification report:\n', c_report)
    
    # Calculating metrics
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy:', round(accuracy*100, 2),'%')
    
    # Area under ROC curve
    roc_auc = roc_auc_score(y_test, y_pred)
    print('roc_auc_score:', round(roc_auc, 3))
    
    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob[:,1])
    plt.figure()
    plt.plot(fpr, tpr, label='KNN (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
new_row.append({'Model Name' : ModelKNN,
               'KNN K Value' : k,
               'True_Positive' : tp,
               'False_Negative' : fn,
               'False_Positive' : fp,
               'True_Negative' : tn,
               'Accuracy' : accuracy,
               'Precision' : precision,
               'Recall' : sensitivity,
               'F1 Score' : f1Score,
               'Specificity' : specificity,
               'MCC':MCC,
               'ROC_AUC_Score':roc_auc_score(y_test, y_pred),
               'Balanced Accuracy':balanced_accuracy})

# Convert the list of dictionaries to a DataFrame
result_df = pd.DataFrame(new_row)

# Append the DataFrame to KNN_Results
KNN_Results = pd.concat([KNN_Results, result_df], ignore_index=True)


# Assuming you have imported and trained a RandomForestClassifier named ModelRF

from matplotlib import pyplot
importance = ModelRF.feature_importances_

# Summarize feature importance
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))

# Plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()


y_pred_r = ModelRF.predict(x_test)


Results = pd.DataFrame({'income_A':y_test, 'income_P':y_pred_r})

# Merge two Dataframes on index of both the dataframes

ResultsFinal = AOdata_bk.merge(Results, left_index=True, right_index=True)
ResultsFinal.sample(20)


#COMPARE WITH ALL OTHER CLASSIFIERS
# Load the results dataset

EMResults = pd.read_csv(r"C:\Users\AMEENA\Downloads\EMResults (2).csv", header=0)

# Display the first 5 records

EMResults.head()


# Build the Calssification models and compare the results

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.naive_bayes import GaussianNB
#from sklearn.svm import SVC
#from xgboost import XGBClassifier
#from sklearn.neural_network import MLPClassifier
#from sklearn.ensemble import GradientBoostingClassifier
#import lightgbm as lgb


# Create objects of classification algorithm with default hyper-parameters

ModelLR = LogisticRegression()
ModelDC = DecisionTreeClassifier()
ModelRF = RandomForestClassifier()
ModelET = ExtraTreesClassifier()
ModelKNN = KNeighborsClassifier(n_neighbors=5)
#ModelGNB = GaussianNB()
#ModelSVM = SVC(probability=True)
#ModelXGB = XGBClassifier(n_estimators=100, max_depth=3, eval_metric='mlogloss')
#ModelMLP = MLPClassifier()
#ModelGB = GradientBoostingClassifier()
#ModelLGB = lgb.LGBMClassifier()

# Evalution matrix for all the algorithms

MM = [ModelLR, ModelDC, ModelRF, ModelET, ModelKNN]

results_list = []

for models in MM:
    
    # Fit the model
    
    models.fit(x_train, y_train)
    
    # Prediction
    
    y_pred = models.predict(x_test)
    y_pred_prob = models.predict_proba(x_test)
    
    # Print the model name
    
    print('Model Name: ', models)
    
    # confusion matrix in sklearn

    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report

    # actual values

    actual = y_test

    # predicted values

    predicted = y_pred

    # confusion matrix

    matrix = confusion_matrix(actual,predicted, labels=[1,0],sample_weight=None, normalize=None)
    print('Confusion matrix : \n', matrix)

    # outcome values order in sklearn

    tp, fn, fp, tn = confusion_matrix(actual,predicted,labels=[1,0]).reshape(-1)
    print('Outcome values : \n', tp, fn, fp, tn)

    # classification report for precision, recall f1-score and accuracy

    C_Report = classification_report(actual,predicted,labels=[1,0])

    print('Classification report : \n', C_Report)

    # calculating the metrics

    sensitivity = round(tp/(tp+fn), 3);
    specificity = round(tn/(tn+fp), 3);
    accuracy = round((tp+tn)/(tp+fp+tn+fn), 3);
    balanced_accuracy = round((sensitivity+specificity)/2, 3);
    
    precision = round(tp/(tp+fp), 3);
    f1Score = round((2*tp/(2*tp + fp + fn)), 3);

    # Matthews Correlation Coefficient (MCC). Range of values of MCC lie between -1 to +1. 
    # A model with a score of +1 is a perfect model and -1 is a poor model

    from math import sqrt

    mx = (tp+fp) * (tp+fn) * (tn+fp) * (tn+fn)
    MCC = round(((tp * tn) - (fp * fn)) / sqrt(mx), 3)

    print('Accuracy :', round(accuracy*100, 2),'%')
    print('Precision :', round(precision*100, 2),'%')
    print('Recall :', round(sensitivity*100,2), '%')
    print('F1 Score :', f1Score)
    print('Specificity or True Negative Rate :', round(specificity*100,2), '%'  )
    print('Balanced Accuracy :', round(balanced_accuracy*100, 2),'%')
    print('MCC :', MCC)

    # Area under ROC curve 

    from sklearn.metrics import roc_curve, roc_auc_score

    print('roc_auc_score:', round(roc_auc_score(actual, predicted), 3))
    
    # ROC Curve
    
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import roc_curve
    model_roc_auc = roc_auc_score(actual, predicted)
    fpr, tpr, thresholds = roc_curve(actual, models.predict_proba(x_test)[:,1])
    plt.figure()
    # plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
    plt.plot(fpr, tpr, label= 'Classification Model' % model_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('Log_ROC')
    plt.show()
    print('-----------------------------------------------------------------------------------------------------')
    #---
   # Append results to the list
    new_row = {'Model Name': str(models),
               'True_Positive': tp,
               'False_Negative': fn,
               'False_Positive': fp,
               'True_Negative': tn,
               'Accuracy': accuracy,
               'Precision': precision,
               'Recall': sensitivity,
               'F1 Score': f1Score,
               'Specificity': specificity,
               'MCC': MCC,
               'ROC_AUC_Score': roc_auc_score(actual, predicted),
               'Balanced Accuracy': balanced_accuracy}
    results_list.append(new_row)

    # --------------------------------------------------------------------------------------------------

# Convert the list to a DataFrame
EMResults = pd.DataFrame(results_list)


#EMResults = pd.read_csv(r"C:\Users\AMEENA\OneDrive\Desktop\DATA ANALYTICS\adult_data.csv", header=0)
EMResults.head(10)
