import pandas as pd
from numpy import arange
from matplotlib import pyplot
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

df=pd.read_excel("C:\\Users\\shobi\\OneDrive\\Desktop\\Anna\\CO2 emission.xlsx")
df=df.iloc[[12,21],[0,50,51,52,53,54,55,56,57,58]]
df=df.set_axis(['year','2006','2007','2008','2009','2010','2011','2012','2013','2014'],axis=1,inplace=False)
df.reset_index(drop=True, inplace=True)
print(df)
df1=df.set_index('year').T
print(df1)

pd.plotting.scatter_matrix(df1 , figsize=(9.0, 9.0))
plt.tight_layout()
plt.show()

kmeans = KMeans(n_clusters=3).fit(df1)
centroids = kmeans.cluster_centers_
print(centroids)


plt.scatter(df1['United Arab Emirates'],df1['Belgium'], c= kmeans.labels_.astype(float), s=200, alpha=0.7)
plt.scatter(centroids[:, 0], centroids[:,1], c='blue', s=200)
plt.title('Clustering')
plt.xlabel('United Arab Emirates')
plt.ylabel('Belgium')
plt.show()

def objective(x,a,b):
    return a*x+b

x ,y = df1['United Arab Emirates'],df1['Belgium']

popt, _ = curve_fit(objective, x, y)
a,b = popt
print('y=%.5f * x+ %.5f' %(a,b))
pyplot.scatter(x,y)
x_line = arange(min(x),max(x),1)
y_line = objective(x_line,a,b)
pyplot.plot(x_line, y_line, color='red', linewidth=1)
plt.xlabel('United Arab Emirates')
plt.ylabel('Belgium')
pyplot.title('fitting')
pyplot.show()
    



