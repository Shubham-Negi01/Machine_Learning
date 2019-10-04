import pandas as pd

df = pd.read_csv("Housing.csv")

X = df["plotsize"]
Y = df["price"]

sum1 = 0
for i in X:
    sum1 += i

mean_X = sum1/ len(X)

for i in Y:
    sum1 += i

mean_Y = sum1 / len(Y)

diff_x = []
diff_y = []
diff_xy = []

for i in X:
    diff_x.append(i - mean_X)
for i in Y:
    diff_y.append(i - mean_Y)

for i in range(len(X)):
    diff_xy.append(diff_x[i]*diff_y[i])

covariance = sum(diff_xy)

variance = 0
for i in diff_x:
    variance+=i**2

slope = covariance/variance


y_intercept = mean_Y - slope*mean_X

df_test = pd.read_csv("Plotsizes.csv")


X_test = df_test["Plotsize"]

Y_predict = []

for i in X_test:
    Y_predict.append(slope*i + y_intercept)


df_test["Prices"] = Y_predict

print(df_test)