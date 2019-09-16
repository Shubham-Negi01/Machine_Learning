import pandas as pd
from sklearn.naive_bayes import GaussianNB
# TRAIN DATA
df = pd.read_csv("train.csv")

df1 = pd.DataFrame()

#removing the columns with large unique value and large null values and assing to new Dataframe df1
df1 = df.drop(["PassengerId", "Age", "Cabin", "Name", "SibSp", "Ticket", "Fare"], axis=1)

# removing the rows with null values
df1.dropna(inplace=True) # "inplace=True" means changes will be saved in df1 itself

#changing the values of columns to integer
sex_list = pd.get_dummies(df1["Sex"])
emb_list = pd.get_dummies(df1["Embarked"])

# dropping the columns with datatype other than integer
df1.drop(["Embarked", "Sex"], axis=1, inplace=True)

# making new DF and appending the new lists with integers
tr = pd.DataFrame()
tr = pd.concat([df1, sex_list, emb_list], axis=1)

# making DF for X-axis
y_train = pd.DataFrame(df1["Survived"])

# making DF for Y-axis
x_train = pd.DataFrame()
x_train = tr.drop(["Survived"],axis =1)


# TEST DATA over

# TEST DATA
df_test=pd.read_csv("test.csv")

df1_test = pd.DataFrame()

df1_test = df_test.drop(["PassengerId", "Age", "Cabin", "Name", "SibSp", "Ticket", "Fare"], axis=1)

df1_test.dropna(inplace=True)

sex_list_test = pd.get_dummies(df1_test["Sex"])
emb_list_test = pd.get_dummies(df1_test["Embarked"])

df1_test.drop(["Embarked", "Sex"], axis=1, inplace=True)

x_test = pd.DataFrame()
x_test = pd.concat([df1_test, sex_list_test, emb_list_test], axis=1)

clf=GaussianNB()
clf.fit(x_train,y_train)
pr=clf.predict(x_test)
print(pr)
