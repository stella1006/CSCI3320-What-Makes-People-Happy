import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.preprocessing import OneHotEncoder


def transform(filename):
    """ preprocess the training data"""
    """ your code here """
    df = pd.read_csv(filename,header=0)
    num_value=df.apply(pd.Series.nunique)
    #print (num_value.loc['YOB'])
    use_one_hot = df[['Income', 'HouseholdStatus', 'EducationLevel', 'Party']];
    df=df.drop('Income',1)
    df=df.drop('HouseholdStatus',1)
    df=df.drop('EducationLevel',1)
    df=df.drop('Party',1)
    df=df.drop('UserID',1)
    use_one_hot=pd.get_dummies(use_one_hot)

    for column in df:

        if column!='YOB' and column!='UserID' and column!='Happy' and column!='votes':
            index=[]
            for val in df[column].unique():
                if isinstance(val, str) or isinstance(val, int):
                    index.append(val)
            format={x:int('%d' % i) for i,x in enumerate(index)}
            df[column]=df[column].map(format)

    result = pd.concat([use_one_hot, df], axis=1)
    # print (result.describe())
    # print (df['YOB'])
    if 'Happy' in list(df.columns.values):
        return {'data':result.drop('Happy',1).as_matrix(),'target':df['Happy'].as_matrix()}
    else:
        return {'data':result.as_matrix()}


def fill_missing(X, strategy, isClassified):
    """
     @X: input matrix with missing data filled by nan
     @strategy: string, 'median', 'mean', 'most_frequent'
     @isclassfied: boolean value, if isclassfied == true, then you need build a
     decision tree to classify users into different classes and use the
     median/mean/mode values of different classes to fill in the missing data;
     otherwise, just take the median/mean/most_frequent values of input data to
     fill in the missing data
    """
    """ your code here """

    data_name=['Income_$100,001 - $150,000', 'Income_$25,001 - $50,000',
                'Income_$50,000 - $74,999', 'Income_$75,000 - $100,000',
                'Income_over $150,000', 'Income_under $25,000', 'HouseholdStatus_Domestic Partners (no kids)',
                'HouseholdStatus_Domestic Partners (w/kids)', 'HouseholdStatus_Married (no kids)',
                'HouseholdStatus_Married (w/kids)', 'HouseholdStatus_Single (no kids)',
                'HouseholdStatus_Single (w/kids)', "EducationLevel_Associate's Degree",
                "EducationLevel_Bachelor's Degree", 'EducationLevel_Current K-12',
                'EducationLevel_Current Undergraduate', 'EducationLevel_Doctoral Degree', 'EducationLevel_High School Diploma',
                "EducationLevel_Master's Degree", 'Party_Democrat', 'Party_Independent',
                'Party_Libertarian', 'Party_Other', 'Party_Republican', 'YOB',
                'Gender', 'Q124742', 'Q124122', 'Q123464', 'Q123621', 'Q122769',
                'Q122770', 'Q122771', 'Q122120', 'Q121699', 'Q121700', 'Q120978', 'Q121011',
                'Q120379', 'Q120650', 'Q120472', 'Q120194', 'Q120012', 'Q120014', 'Q119334',
                'Q119851', 'Q119650', 'Q118892', 'Q118117', 'Q118232', 'Q118233', 'Q118237',
                'Q117186', 'Q117193', 'Q116797', 'Q116881', 'Q116953', 'Q116601', 'Q116441',
                'Q116448', 'Q116197', 'Q115602', 'Q115777', 'Q115610', 'Q115611', 'Q115899',
                'Q115390', 'Q114961', 'Q114748', 'Q115195', 'Q114517', 'Q114386', 'Q113992',
                'Q114152', 'Q113583', 'Q113584', 'Q113181', 'Q112478', 'Q112512', 'Q112270',
                'Q111848', 'Q111580', 'Q111220', 'Q110740', 'Q109367', 'Q108950', 'Q109244',
                'Q108855', 'Q108617', 'Q108856', 'Q108754', 'Q108342', 'Q108343', 'Q107869',
                'Q107491', 'Q106993', 'Q106997', 'Q106272', 'Q106388', 'Q106389', 'Q106042',
                'Q105840', 'Q105655', 'Q104996', 'Q103293', 'Q102906', 'Q102674', 'Q102687',
                'Q102289', 'Q102089', 'Q101162', 'Q101163', 'Q101596', 'Q100689', 'Q100680',
                'Q100562', 'Q99982', 'Q100010', 'Q99716', 'Q99581', 'Q99480', 'Q98869', 'Q98578',
                'Q98059', 'Q98078', 'Q98197', 'Q96024', 'votes']

    # print (data_name.index('Happy'))
    print (len(data_name),X.shape[1])
    if len(data_name)!=X.shape[1] :
        data_name.remove('Happy')

    X=pd.DataFrame(X,columns=data_name)

    # print (X)
    if isClassified==False:
        if strategy=='most_frequent':
            for column in X:
                if column!='YOB' and column!='UserID' and column!='Happy' and column!='votes':
                    #print (column, df[column].unique())
                    most_fre = X[column].mode()
                    X[column]=X[column].fillna('%d'% most_fre)
                elif column=='YOB':
                    most_fre = X['YOB'].mode()
                    X['YOB']=X['YOB'].fillna(most_fre.max())

        elif strategy=='mean':
            for column in X:
                if column!='YOB' and column!='UserID' and column!='Happy' and column!='votes':
                    mean_val=X[column].mean()
                    X[column]=X[column].fillna('%d'%round(mean_val))
                elif column=='YOB':
                    mean_val = X['YOB'].mean()
                    X['YOB']=X['YOB'].fillna(mean_val)
        else:
            for column in X:
                if column!='YOB' and column!='UserID' and column!='Happy' and column!='votes':
                    #print (column, df[column].unique())
                    median_val = round(X[column].median())
                    X[column]=X[column].fillna('%d'%median_val)
                elif column=='YOB':
                    median_val = X['YOB'].median()
                    X['YOB']=X['YOB'].fillna(median_val)


    else:
        X_missing = X.dropna()
        print ("Filling missing data starts! Please pay attention to wait a minute!")
        names=list(X.columns.values)
        names.remove('votes')
        if 'Happy' in names:
            names.remove('Happy')
        # names.remove('UserID')

        for name in names:
            target=X_missing[name]
            X_missing = X_missing.drop(name,1)
            clf = tree.DecisionTreeClassifier(max_depth=2)
            clf=clf.fit(X_missing,target)
            names_missing=list(X_missing.columns.values)
            indexs=[]
            for i,x in enumerate(clf.feature_importances_):
                if x>0:
                    indexs.append(i)
            if len(indexs)==0:
                indexs=[0,1]
            indexs = sorted(indexs, key=int, reverse=True);

            attri=[]
            for i in indexs[0:3]:
                if i < len(names_missing):
                    attri.append(names_missing[i])

            X_copy = X.copy()
            grouped=X_copy.groupby(attri)
            count=0
            for na, group in grouped:
                if strategy=='mean':
                    mean_val = group[name].mean()
                elif strategy=='median':
                    mean_val = group[name].median()
                elif strategy=='most_frequent':
                    mean_val = group[name].mode().max()

                if np.isnan((mean_val))==True:
                    mean_val = np.nan
                else:
                    mean_val = round(mean_val)

                indexs=grouped.indices[na]

                for index in indexs:
                    if np.isnan(X.loc[index][name])==True:
                        count+=1
                        X.set_value(index,name,mean_val)

        print (count)
        X_after=X.dropna()
        print (X_after.as_matrix().shape)
        print (X.shape)
    if isClassified==True:
        X = fill_missing(X.as_matrix(), 'most_frequent', False)
        return X
    else:
        return X.as_matrix()


# filename_train = './data/train.csv'
# train_dataset = transform(filename_train)
# X = train_dataset['data']
# y = train_dataset['target']

# fill in missing data (optional)
# X = fill_missing(X, 'mean', F)
# X_full = fill_missing(X, 'mean', True)
# X_full.as_matrix()
# print (X_full.shape)
