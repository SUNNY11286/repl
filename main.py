import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression  
from sklearn.metrics import mean_squared_error, r2_score

# Importing the dataset
dta = pd.read_csv('Spring_data.csv')
#dta = pd.read_csv('salary_data.csv')
print(dta)

x=dta.iloc[:,:-1].values
y=dta.iloc[:,1].values
print('x = ',x)
#dt = dta[dta.columns[:]].corr()[y][:]
#print(dt)

'''min-max normalization'''
for column in dta.columns: 
    if dta[column].dtype == 'object':
        continue
    dta[column] = (dta[column] - dta[column].min()) / (dta[column].max() - dta[column].min())     
print(dta)

dt = dta.corr()
print(dt)

x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 1/3, random_state=0)  
reg= LinearRegression()  
reg.fit(x_train, y_train)  
#Prediction of Test and Training set result  
y_pred= reg.predict(x_test)  
x_pred= reg.predict(x_train)  

print('x_test',x_test,'y-preds', y_pred)

rmse =  sqrt(mean_squared_error(y_pred, y_test))
r2=r2_score(y_test,y_pred)
print(rmse)
print(r2)

plt.figure()
plt.title('spring displaced due to mass')
plt.xlabel('displacement of spring')
plt.ylabel('mass')
#plt.scatter(x,y, color='blue')
#plt.scatter(x_train,y_train, color='blue')
plt.scatter(x_test,y_test, color='blue')
plt.plot(x_train,x_pred,'-k')
plt.show()

y_pred5=reg.predict(x['5.0'])
print('for 5 =',y_pred5)