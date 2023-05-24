#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# In[11]:


iris = pd.read_csv('file:///C:/Users/hp/Desktop/iris.csv')
iris


# In[12]:


iris.shape


# In[13]:


iris.columns


# In[14]:


iris['species'].value_counts()


# In[15]:


iris.info()


# In[16]:


versicolor = iris.loc[iris['species'] == "versicolor"]
setosa = iris.loc[iris['species'] == "setosa"]
virginica = iris.loc[iris['species'] == "virginica"]


# # 1.1 D-Scatter Plots

# In[20]:


plt.figure(1)
plt.plot(versicolor["sepal_length"], np.zeros_like(versicolor['sepal_length']), 'o', label = 'versicolor')
plt.plot(setosa["sepal_length"], np.zeros_like(setosa['sepal_length']), 'o', label = 'setosa')
plt.plot(virginica["sepal_length"], np.zeros_like(virginica['sepal_length']), 'o', label = 'virginica')
plt.title("1-D Scatter plot of sepal_length")
plt.xlabel("sepal_length")
plt.legend()
plt.show()


# # 1.2 Histogram and PDF

# In[29]:


sns.FacetGrid(iris, hue = "species").map(sns.distplot, "petal_length").add_legend();
plt.title("Histogram of petal_length")
plt.ylabel("probability Density of petal_length")
plt.show


# In[31]:


sns.FacetGrid(iris, hue = "species").map(sns.distplot, "petal_width").add_legend();
plt.title("Histogram of petal_width")
plt.ylabel("probability Density of petal_width")
plt.show


# In[32]:


sns.FacetGrid(iris, hue = "species").map(sns.distplot, "sepal_length").add_legend();
plt.title("Histogram of sepal_length")
plt.ylabel("probability Density of sepal_length")
plt.show


# In[33]:


sns.FacetGrid(iris, hue = "species").map(sns.distplot, "sepal_width").add_legend();
plt.title("Histogram of sepal_width")
plt.ylabel("probability Density of sepal_width")
plt.show


# # 1.3 CDF (Cumulative distribution function):

# In[38]:


plt.figure(1)
counts, bin_edges = np.histogram(versicolor['sepal_length'], bins=10, density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf, label = "pdf of versicolor")
plt.plot(bin_edges[1:], cdf, label = "cdf of versicolor")
plt.xlabel("sepal_length")
plt.ylabel("cummulative probability density")
plt.title("CDF of sepal_length for versicolor")
plt.legend()


# In[39]:


plt.figure(4)
counts, bin_edges = np.histogram(versicolor['sepal_length'], bins=10, density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf, label = "pdf of versicolor")
plt.plot(bin_edges[1:], cdf, label = "cdf of versicolor")

counts, bin_edges = np.histogram(setosa['sepal_length'], bins=20, density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf, label = "pdf of setosa")
plt.plot(bin_edges[1:], cdf, label = "cdf of setosa")

counts, bin_edges = np.histogram(virginica['sepal_length'], bins=20, density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf, label = "pdf of virginica")
plt.plot(bin_edges[1:], cdf, label = "cdf of virginica")


# In[40]:


#from other features

plt.figure(4)
counts, bin_edges = np.histogram(versicolor['sepal_width'], bins=10, density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf, label = "pdf of versicolor")
plt.plot(bin_edges[1:], cdf, label = "cdf of versicolor")

counts, bin_edges = np.histogram(setosa['sepal_width'], bins=20, density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf, label = "pdf of setosa")
plt.plot(bin_edges[1:], cdf, label = "cdf of setosa")

counts, bin_edges = np.histogram(virginica['sepal_width'], bins=20, density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf, label = "pdf of virginica")
plt.plot(bin_edges[1:], cdf, label = "cdf of virginica")


# In[41]:


#from other features

plt.figure(4)
counts, bin_edges = np.histogram(versicolor['petal_length'], bins=10, density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf, label = "pdf of versicolor")
plt.plot(bin_edges[1:], cdf, label = "cdf of versicolor")

counts, bin_edges = np.histogram(setosa['petal_length'], bins=20, density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf, label = "pdf of setosa")
plt.plot(bin_edges[1:], cdf, label = "cdf of setosa")

counts, bin_edges = np.histogram(virginica['petal_length'], bins=20, density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf, label = "pdf of virginica")
plt.plot(bin_edges[1:], cdf, label = "cdf of virginica")


# In[42]:


#from other features

plt.figure(4)
counts, bin_edges = np.histogram(versicolor['petal_width'], bins=10, density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf, label = "pdf of versicolor")
plt.plot(bin_edges[1:], cdf, label = "cdf of versicolor")

counts, bin_edges = np.histogram(setosa['petal_width'], bins=20, density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf, label = "pdf of setosa")
plt.plot(bin_edges[1:], cdf, label = "cdf of setosa")

counts, bin_edges = np.histogram(virginica['petal_width'], bins=20, density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf, label = "pdf of virginica")
plt.plot(bin_edges[1:], cdf, label = "cdf of virginica")


# # 2. Statistical analysis using Mean, Median, STD, Quantiles and Percentiles:

# In[53]:


print("Means : ")
print("Mean of sepal_length for versicolor = ", np.mean(versicolor['sepal_length']))
print("Mean of sepal_length for setosa = ", np.mean(setosa['sepal_length']))
print("Mean of sepal_length for virginica = ", np.mean(virginica['sepal_length']))
print("____________________________________________________________________________")
print("Mean of sepal_width for versicolor = ", np.mean(versicolor['sepal_width']))
print("Mean of sepal_width for setosa = ", np.mean(setosa['sepal_width']))
print("Mean of sepal_width for virginica = ", np.mean(virginica['sepal_width']))
print("____________________________________________________________________________")
print("Mean of petal_length for versicolor = ", np.mean(versicolor['petal_length']))
print("Mean of petal_length for setosa = ", np.mean(setosa['petal_length']))
print("Mean of petal_length for virginica = ", np.mean(virginica['petal_length']))
print("____________________________________________________________________________")
print("Mean of petal_width for versicolor = ", np.mean(versicolor['petal_width']))
print("Mean of petal_width for setosa = ", np.mean(setosa['petal_width']))
print("Mean of petal_width for virginica = ", np.mean(virginica['petal_width']))

print("std-dev : ")
print("std-dev of sepal_length for versicolor = ", np.std(versicolor['sepal_length']))
print("std-dev of sepal_length for setosa = ", np.std(setosa['sepal_length']))
print("std-dev of sepal_length for virginica = ", np.std(virginica['sepal_length']))
print("____________________________________________________________________________")
print("std-dev of sepal_width for versicolor = ", np.std(versicolor['sepal_width']))
print("std-dev of sepal_width for setosa = ", np.std(setosa['sepal_width']))
print("std-dev of sepal_width for virginica = ", np.std(virginica['sepal_width']))
print("____________________________________________________________________________")
print("std-dev of petal_length for versicolor = ", np.std(versicolor['petal_length']))
print("std-dev of petal_length for setosa = ", np.std(setosa['petal_length']))
print("std-dev of petal_length for virginica = ", np.std(virginica['petal_length']))
print("____________________________________________________________________________")
print("std-dev of petal_width for versicolor = ", np.std(versicolor['petal_width']))
print("std-dev of petal_width for setosa = ", np.std(setosa['petal_width']))
print("std-dev of petal_width for virginica = ", np.std(virginica['petal_width']))


# # 2.2 Median, Percentile, Quantile, IQR, MAD:

# In[55]:


print("Median : ")
print("Median of sepal_length for versicolor = ", np.median(versicolor['sepal_length']))
print("Median of sepal_length for setosa = ", np.median(setosa['sepal_length']))
print("Median of sepal_length for virginica = ", np.median(virginica['sepal_length']))
print("____________________________________________________________________________")
print("Median of sepal_width for versicolor = ", np.median(versicolor['sepal_width']))
print("Median of sepal_width for setosa = ", np.median(setosa['sepal_width']))
print("Median of sepal_width for virginica = ", np.median(virginica['sepal_width']))
print("____________________________________________________________________________")
print("Median of petal_length for versicolor = ", np.median(versicolor['petal_length']))
print("Median of petal_length for setosa = ", np.median(setosa['petal_length']))
print("Median of petal_length for virginica = ", np.median(virginica['petal_length']))
print("____________________________________________________________________________")
print("Median of petal_width for versicolor = ", np.median(versicolor['petal_width']))
print("Median of petal_width for setosa = ", np.median(setosa['petal_width']))
print("Median of petal_width for virginica = ", np.median(virginica['petal_width']))

print("\n\n Quantiles : ")
print("Quantiles of sepal_length for versicolor = ", np.percentile(versicolor['sepal_length'], np.arange(0, 100, 25)))
print("Quantiles of sepal_length for setosa = ", np.percentile(setosa['sepal_length'], np.arange(0, 100, 25)))
print("Quantiles of sepal_length for virginica = ", np.percentile(virginica['sepal_length'], np.arange(0, 100, 25)))
print("____________________________________________________________________________")
print("Quantiles of sepal_width for versicolor = ", np.percentile(versicolor['sepal_width'], np.arange(0, 100, 25)))
print("Quantiles of sepal_width for setosa = ", np.percentile(setosa['sepal_width'], np.arange(0, 100, 25)))
print("Quantiles of sepal_width for virginica = ", np.percentile(virginica['sepal_width'], np.arange(0, 100, 25)))
print("____________________________________________________________________________")
print("Quantiles of petal_length for versicolor = ", np.percentile(versicolor['petal_length'], np.arange(0, 100, 25)))
print("Quantiles of petal_length for setosa = ", np.percentile(setosa['petal_length'], np.arange(0, 100, 25)))
print("Quantiles of petal_length for virginica = ", np.percentile(virginica['petal_length'], np.arange(0, 100, 25)))
print("____________________________________________________________________________")
print("Quantiles of petal_width for versicolor = ", np.percentile(versicolor['petal_width'], np.arange(0, 100, 25)))
print("Quantiles of petal_width for setosa = ", np.percentile(setosa['petal_width'], np.arange(0, 100, 25)))
print("Quantiles of petal_width for virginica = ", np.percentile(virginica['petal_width'], np.arange(0, 100, 25)))


print("\n\n 90th Percentile : ")
print("90th Percentile of sepal_length for versicolor = ", np.percentile(versicolor['sepal_length'], 90))
print("90th Percentile of sepal_length for setosa = ", np.percentile(setosa['sepal_length'], 90))
print("90th Percentile of sepal_length for virginica = ", np.percentile(virginica['sepal_length'], 90))
print("____________________________________________________________________________")
print("90th Percentile of sepal_width for versicolor = ", np.percentile(versicolor['sepal_width'], 90))
print("90th Percentile of sepal_width for setosa = ", np.percentile(setosa['sepal_width'], 90))
print("90th Percentile of sepal_width for virginica = ", np.percentile(virginica['sepal_width'], 90))
print("____________________________________________________________________________")
print("90th Percentile of petal_length for versicolor = ", np.percentile(versicolor['petal_length'], 90))
print("90th Percentile of petal_length for setosa = ", np.percentile(setosa['petal_length'], 90))
print("90th Percentile of petal_length for virginica = ", np.percentile(virginica['petal_length'], 90))
print("____________________________________________________________________________")
print("90th Percentile of petal_width for versicolor = ", np.percentile(versicolor['petal_width'], 90))
print("90th Percentile of petal_width for setosa = ", np.percentile(setosa['petal_width'], 90))
print("90th Percentile of petal_width for virginica = ", np.percentile(virginica['petal_width'], 90))

from statsmodels import robust

print("Median Absolute Deviation : ")
print("Median Absolute Deviation of sepal_length for versicolor = ", robust.mad(versicolor['sepal_length']))
print("Median Absolute Deviation of sepal_length for setosa = ", robust.mad(setosa['sepal_length']))
print("Median Absolute Deviation of sepal_length for virginica = ", robust.mad(virginica['sepal_length']))
print("____________________________________________________________________________")
print("Median Absolute Deviation of sepal_width for versicolor = ", robust.mad(versicolor['sepal_width']))
print("Median Absolute Deviation of sepal_width for setosa = ", robust.mad(setosa['sepal_width']))
print("Median Absolute Deviation of sepal_width for virginica = ", robust.mad(virginica['sepal_width']))
print("____________________________________________________________________________")
print("Median Absolute Deviation of petal_length for versicolor = ", robust.mad(versicolor['petal_length']))
print("Median Absolute Deviation of petal_length for setosa = ", robust.mad(setosa['petal_length']))
print("Median Absolute Deviation of petal_length for virginica = ", robust.mad(virginica['petal_length']))
print("____________________________________________________________________________")
print("Median Absolute Deviation of petal_width for versicolor = ", robust.mad(versicolor['petal_width']))
print("Median Absolute Deviation of petal_width for setosa = ", robust.mad(setosa['petal_width']))
print("Median Absolute Deviation of petal_width for virginica = ", robust.mad(virginica['petal_width']))


# # 3. Univariate analysis with BOX plots and Violin plots:

# # 3.1 Box plot and Whiskers

# In[58]:


#Box plot is drawn for visualizing percentile, quantile

sns.boxplot(x='species', y='sepal_length', data=iris)
plt.title("BOX Plot for sepal_length")
plt.show()

sns.boxplot(x='species', y='sepal_width', data=iris)
plt.title("BOX Plot for sepal_width")
plt.show()

sns.boxplot(x='species', y='petal_length', data=iris)
plt.title("BOX Plot for pegal_length")
plt.show()

sns.boxplot(x='species', y='petal_width', data=iris)
plt.title("BOX Plot for petal_width")
plt.show()


# # 3.2 Violin plots:

# In[62]:


#violin plot conbines the advantages of Box plot and PDFs

sns.violinplot(x='species', y='sepal_length', data=iris, size=8)
plt.title("BOX Plot for sepal_length")
plt.show()

sns.violinplot(x='species', y='sepal_width', data=iris, size=8)
plt.title("BOX Plot for sepal_width")
plt.show()

sns.violinplot(x='species', y='sepal_width', data=iris, size=8)
plt.title("BOX Plot for sepal_width")
plt.show()

sns.violinplot(x='species', y='petal_width', data=iris, size=8)
plt.title("BOX Plot for petal_width")
plt.show()


# # 4. Bi-Variate analysis / Multivariate analysis:

# # 4.1 2-D Scatter Plots: Scatter plot

# In[65]:


sns.set_style("whitegrid")
sns.FacetGrid(iris, hue='species').map(plt.scatter, "sepal_length", "sepal_width").add_legend();
plt.title("2-D Scatter plot for sepal_length, sepal_width")
plt.show()


# # 4.2 Pair-plot (To observe all 2-D Scatter Plot):

# In[70]:


sns.set_style("whitegrid")
sns.pairplot(iris, hue='species', vars=['sepal_length', 'sepal_width', 'petal_width', 'petal_width'], height=3);
plt.title("Pairplot")
plt.show()


# # 5. Multivariate probability density plot or Contour Plot

# In[72]:


sns.jointplot(x='sepal_length', y='sepal_width', data=setosa, kind="kde")
plt.show()


# In[74]:


sns.jointplot(x='sepal_length', y='sepal_width', data=virginica, kind="kde")
plt.show()


# In[75]:


sns.jointplot(x='sepal_length', y='sepal_width', data=versicolor, kind="kde")
plt.show()


# In[ ]:




