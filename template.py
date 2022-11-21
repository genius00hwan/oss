# PLEASE WRITE THE GITHUB URL BELOW!
#https://github.com/genius00hwan/oss.git
import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def load_dataset(dataset_path):
    data_df = pd.read_csv(dataset_path)
    return data_df


# To-Do: Implement this function

def dataset_stat(dataset_df):
    nof = len(dataset_df.columns) - 1
    no0 = len(dataset_df.loc[dataset_df["target"] == 0])
    no1 = len(dataset_df.loc[dataset_df["target"] == 1])
    return nof, no0, no1


# To-Do: Implement this function

def split_dataset(dataset_df, testset_size):
    X = dataset_df.drop(columns="target", axis=1)
    y = dataset_df["target"]
    xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=testset_size)
    return xTrain, xTest, yTrain, yTest


# To-Do: Implement this function

def decision_tree_train_test(x_train, x_test, y_train, y_test):
    dt_cls = DecisionTreeClassifier()
    dt_cls.fit(x_train, y_train)
    dt_acc = accuracy_score(y_test, dt_cls.predict(x_test))
    dt_pre = precision_score(y_test, dt_cls.predict(x_test))
    dt_rec = recall_score(y_test, dt_cls.predict(x_test))
    return dt_acc, dt_pre, dt_rec


def random_forest_train_test(x_train, x_test, y_train, y_test):
    rf_cls = RandomForestClassifier()
    rf_cls.fit(x_train, y_train)
    rf_acc = accuracy_score(y_test,rf_cls.predict(x_test))
    rf_pre = precision_score(y_test,rf_cls.predict(x_test))
    rf_rec = recall_score(y_test,rf_cls.predict(x_test))
    return rf_acc, rf_pre, rf_rec


#
def svm_train_test(x_train, x_test, y_train, y_test):
    svm_pipe = make_pipeline(
        StandardScaler(),
        SVC()
    )
    svm_pipe.fit(x_train, y_train)
    svm_acc = accuracy_score(y_test, svm_pipe.predict(x_test))
    svm_pre = precision_score(y_test, svm_pipe.predict(x_test))
    svm_rec = recall_score(y_test, svm_pipe.predict(x_test))
    return svm_acc, svm_pre, svm_rec

def print_performances(acc, prec, recall):
    # Do not modify this function!
    print("Accuracy: ", acc)
    print("Precision: ", prec)
    print("Recall: ", recall)


if __name__ == '__main__':
    # Do not modify the main script!
    data_path = sys.argv[1]
    data_df = load_dataset(data_path)

    n_feats, n_class0, n_class1 = dataset_stat(data_df)
    print("Number of features: ", n_feats)
    print("Number of class 0 data entries: ", n_class0)
    print("Number of class 1 data entries: ", n_class1)

    print("\nSplitting the dataset with the test size of ", float(sys.argv[2]))
    x_train, x_test, y_train, y_test = split_dataset(data_df, float(sys.argv[2]))

    acc, prec, recall = decision_tree_train_test(x_train, x_test, y_train, y_test)
    print("\nDecision Tree Performances")
    print_performances(acc, prec, recall)

    acc, prec, recall = random_forest_train_test(x_train, x_test, y_train, y_test)
    print("\nRandom Forest Performances")
    print_performances(acc, prec, recall)

    acc, prec, recall = svm_train_test(x_train, x_test, y_train, y_test)
    print("\nSVM Performances")
    print_performances(acc, prec, recall)
