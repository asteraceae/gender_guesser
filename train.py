#In this class we shall train a model for use

from sklearn import svm
import pandas as pd


with open("data.csv", "r+", errors = 'ignore') as f:
    #df = f.readlines()
    df = pd.read_csv(f, delimiter = ",")

#Because this csv is being a pain-
#Columns: Gender, Gender Confidence, Description, Tweet Text


print(df["text"])



#model = svm.SVC()
#model.fit(x_train, y_train)