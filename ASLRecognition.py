
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier , plot_tree
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import Testings as ts


from sklearn.svm import SVC

def ASL_Model_training():
    data_combined = pd.DataFrame()
    CSV_data = Path("ASL Dataset/CSV_landmarks_AL_2").rglob('*.csv')
    files = [x for x in CSV_data]
    for i in files:
        newData = pd.read_csv(i)
        data_combined = pd.concat([data_combined ,newData])
    training_data = data_combined.drop("Class", axis=1)


    classes = data_combined["Class"].unique()
    #classe = data_combined['Class']

    #knn = RandomForestClassifier(verbose=2)
    #sfs = SequentialFeatureSelector(knn , n_jobs=2)

    #print(sfs.fit(training_data,classe))

    #old_data = training_data
    #x_new = sfs.transform(training_data)

    #print(sfs)
    #print(training_data.shape)
    #print(x_new.shape)

    Xtrain, Xtest, ytrain, ytest = train_test_split(training_data, data_combined["Class"], test_size=0.3)
    #model = RandomForestClassifier(n_jobs=-1)
    #model.fit(Xtrain, ytrain)
    model2 = DecisionTreeClassifier(max_depth=1)
    model2.fit(Xtrain, ytrain)

    plot_tree(model2)
    plt.show()


    #model3 = GaussianNB()
    #model3.fit(Xtrain, ytrain)
    print("passed")
    #%%
    #ts.Model_Testing(model, Xtest, ytest, classes, Xtrain, ytrain, training_data, data_combined["Class"])
    #filename =
    filename = 'self_Model.sav'
    pickle.dump(model, open("./images/" + filename, 'wb'))

