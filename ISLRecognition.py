import pandas as pd
from pathlib import Path
from sklearn.naive_bayes import GaussianNB
import Testings as ts
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pickle
import Testings as ts
from sklearn.metrics import accuracy_score, confusion_matrix , precision_score , f1_score , recall_score


from sklearn.ensemble import RandomForestClassifier


def ISL_Model_training():
    data_combined = pd.DataFrame()
    CSV_data = Path("ISL Dataset/landmarks_with_lines_2").rglob('*.csv')

    files = [x for x in CSV_data]
    for i in files:
        newData = pd.read_csv(i)
        newData = newData.drop([0])
        data_combined = pd.concat([data_combined ,newData])

    classes = data_combined["Class"].unique()

    training_data = data_combined.drop("Class", axis=1)

    print(training_data.shape)

    correlated_features = set()
    correlation_matrix = training_data.corr()
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > 0.9:
                colname = correlation_matrix.columns[i]
                correlated_features.add(colname)
    training_data.drop(labels=correlated_features, axis=1, inplace=True)

    print(training_data.shape)

    df_max_scaled = training_data.copy()

    # apply normalization techniques
    for column in df_max_scaled.columns:
        df_max_scaled[column] = df_max_scaled[column] / df_max_scaled[column].abs().max()



    Xtrain, Xtest, ytrain, ytest = train_test_split(df_max_scaled, data_combined["Class"], test_size=0.3 , random_state=10)
    model = RandomForestClassifier()
    model.fit(Xtrain, ytrain)
    print("passed")
    ts.Model_Testing(model, Xtest, ytest, classes, Xtrain, ytrain, training_data,
                     data_combined["Class"])
    #filename = 'ISL Dataset/Model/Model_RandomForest_AL_2.sav'
    #pickle.dump(model, open(filename, 'wb'))

    #ts.Model_Testing(model , Xtest , ytest , classes , Xtrain , ytrain , training_data , data_combined["Class"])
