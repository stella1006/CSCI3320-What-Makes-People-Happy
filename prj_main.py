from preprocess import transform
from preprocess import fill_missing
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
import numpy as np
import pandas as pd


def main():
    # load training data
    filename_train = "./data/train.csv"
    filename_test = "./data/test.csv"
    df = pd.read_csv(filename_test,header=0)
    X_pre_userId=df['UserID']
    X_pre_userId=X_pre_userId.as_matrix()
    train_dataset = transform(filename_train)
    test_dateset = transform(filename_test)

    X = train_dataset['data']
    y = train_dataset['target']
    X_pre = test_dateset['data']
    num_train=X.shape[0]
    X=np.append(X,X_pre,0)

    X_fill = fill_missing(X, 'most_frequent', False)
    # X_fill = fill_missing(X, 'most_frequent', True)
    X_pre_fill=X_fill[num_train::]
    X_fill=X_fill[0:num_train]

    X_train, X_test, y_train, y_test = train_test_split(X_fill, y, test_size=0.2, random_state=4)
    print (y_train.shape,y_test.shape)


    ### use the logistic regression
    print('Train the logistic regression classifier')
    """ your code here """
    lr_model = LogisticRegression(random_state=4)
    lr_model.fit(X_train, y_train)
    print(lr_model.score(X_test,y_test))
    lr_pre = lr_model.predict(X_pre_fill)
    file = open('./predictions/lr_predictions.csv','w')
    file.write('UserID,Happy\n')
    for temp in range(0,lr_pre.shape[0]):
        file.write('%d'%X_pre_userId[temp])
        file.write(',')
        file.write(str(lr_pre[temp]))
        file.write('\n')

    ### use the naive bayes
    print('Train the naive bayes classifier')
    """ your code here """
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    print(nb_model.score(X_test,y_test))
    nb_pre = nb_model.predict(X_pre_fill)
    file = open('./predictions/nb_predictions.csv','w')
    file.write('UserID,Happy\n')
    for temp in range(0,nb_pre.shape[0]):
        file.write('%d'%X_pre_userId[temp])
        file.write(',')
        file.write(str(nb_pre[temp]))
        file.write('\n')

    ## use the svm
    print('Train the SVM classifier')
    """ your code here """
    svm_model = svm.SVC(kernel='linear', random_state=0)
    svm_model.fit(X_train, y_train)
    print(svm_model.score(X_test,y_test))
    svm_pre = svm_model.predict(X_pre_fill)
    file = open('./predictions/svm_predictions.csv','w')
    file.write('UserID,Happy\n')
    for temp in range(0,svm_pre.shape[0]):
        file.write('%d'%X_pre_userId[temp])
        file.write(',')
        file.write(str(svm_pre[temp]))
        file.write('\n')

    ## use the random forest
    print('Train the random forest classifier')
    """ your code here """
    rf_model = RandomForestClassifier(n_estimators=2600,random_state=4)
    rf_model = rf_model.fit(X_train, y_train)
    print(rf_model.score(X_test,y_test))
    rf_pre = rf_model.predict(X_pre_fill)
    file = open('./predictions/rf_predictions.csv','w')
    file.write('UserID,Happy\n')
    for temp in range(0,rf_pre.shape[0]):
        file.write('%d'%X_pre_userId[temp])
        file.write(',')
        file.write(str(rf_pre[temp]))
        file.write('\n')

    ## get predictions
    """ your code here """

if __name__ == '__main__':
    main()
