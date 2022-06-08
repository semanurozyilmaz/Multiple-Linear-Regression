import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import r2_score
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

df = pd.read_csv("AdmissionPredict_mlr.csv")
df.drop('SerialNo',axis=1, inplace=True)
print(df)

x = df.iloc[:,0:7].values
y = df.iloc[:,7:].values

sns.heatmap(df.corr(),annot=True,linewidths=0.5,fmt='.2f')

plt.scatter(df.GREScore,df.ChanceofAdmit)
plt.xlabel("GRE Score")
plt.ylabel("Chance of Admit")
plt.title("GRE Score - Chance of Admit Relationship")
plt.grid(True)
plt.show()

plt.scatter(df.TOEFLScore,df.ChanceofAdmit)
plt.xlabel("TOEFL Score")
plt.ylabel("Chance of Admit")
plt.title("TOEFL Score - Chance of Admit Relationship")
plt.grid(True)
plt.show()

plt.scatter(df.UniversityRating,df.ChanceofAdmit)
plt.xlabel("UniversityRating")
plt.ylabel("Chance of Admit")
plt.title("UniversityRating - Chance of Admit Relationship")
plt.grid(True)
plt.show()

plt.scatter(df.SOP,df.ChanceofAdmit)
plt.xlabel("SOP")
plt.ylabel("Chance of Admit")
plt.title("SOP - Chance of Admit Relationship")
plt.grid(True)
plt.show()

plt.scatter(df.LOR,df.ChanceofAdmit)
plt.xlabel("LOR")
plt.ylabel("Chance of Admit")
plt.title("LOR - Chance of Admit Relationship")
plt.grid(True)
plt.show()

plt.scatter(df.CGPA,df.ChanceofAdmit)
plt.xlabel("CGPA")
plt.ylabel("Chance of Admit")
plt.title("CGPA - Chance of Admit Relationship")
plt.grid(True)
plt.show()

plt.scatter(df.Research,df.ChanceofAdmit)
plt.xlabel("Research")
plt.ylabel("Chance of Admit")
plt.title("Research - Chance of Admit Relationship")
plt.grid(True)
plt.show()

mlr = LinearRegression()
mlr.fit(x,y)

model=sm.OLS(mlr.predict(x),x)
print(model.fit().summary())

print('Linear R2 degeri')
print(r2_score(y, mlr.predict(x)))