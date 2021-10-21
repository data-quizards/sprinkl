#!/usr/bin/env python
# coding: utf-8

# In[1]:


from glob import glob

import earthpy as et
import earthpy.spatial as es
import earthpy.plot as ep

import rasterio as rio
from rasterio.plot import show
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

import plotly.graph_objects as go

from rasterio.plot import plotting_extent

np.seterr(divide='ignore', invalid='ignore')


# In[2]:


landsat_bands1 = glob("Jan 2021/////B?*.png")
landsat_bands1.sort()

m = []
for i in landsat_bands1:
    with rio.open(i, 'r') as f:
        m.append(f.read(1))

# Data
arr_st1 = np.stack(m)


# In[3]:


ep.plot_bands(arr_st1, cmap = 'gist_earth', figsize = (20, 12), cols = 6, cbar = False)
plt.show()


# In[4]:


ep.plot_rgb(
    arr_st1,
    rgb=(4,3,2),
    stretch=False,
    str_clip=0.2,
    figsize=(10, 16),
    title="RGB Image with Stretch Applied",
)

plt.show()


# In[5]:


colors = ['red', 'green', 'orange', 'blue',
          'black', 'purple', 'yellow',  'brown']

ep.hist(arr_st1, 
         colors = colors,
        title=[f'Band-{i}' for i in range(1, 9)], 
        cols=3, 
        alpha=0.5, 
        figsize = (12, 10)
        )

plt.show()


# ## Normalized Difference Moisture Index (NDMI)

# # JAN 2021

# In[6]:


ndmi1 = es.normalized_diff(arr_st1[5], arr_st1[6])

ep.plot_bands(ndmi1, cmap="RdYlGn", cols=1, vmin=-1, vmax=1, figsize=(10, 14))

plt.show()


# In[7]:


len(ndmi1[ndmi1 > 0.4])


# # FEB 2021

# In[8]:


landsat_bands2 = glob("Feb 2021/////B?*.png")
landsat_bands2.sort()

n = []
for i in landsat_bands2:
    with rio.open(i, 'r') as f:
        n.append(f.read(1))

# Data
arr_st2 = np.stack(n)


# In[9]:


ep.plot_bands(arr_st2, cmap = 'gist_earth', figsize = (20, 12), cols = 6, cbar = False)
plt.show()


# In[10]:


ndmi2 = es.normalized_diff(arr_st2[5], arr_st2[6])

ep.plot_bands(ndmi2, cmap="RdYlGn", cols=1, vmin=-1, vmax=1, figsize=(10, 14))

plt.show()


# In[11]:


len(ndmi2[ndmi2 > 0.4])


# # MAR 2021

# In[12]:


landsat_bands3 = glob("Mar 2021///B?*.png")
landsat_bands3.sort()

o = []
for i in landsat_bands3:
    with rio.open(i, 'r') as f:
        o.append(f.read(1))

# Data
arr_st3 = np.stack(o)


# In[13]:


ep.plot_bands(arr_st3, cmap = 'gist_earth', figsize = (20, 12), cols = 6, cbar = False)
plt.show()


# In[14]:


ndmi3 = es.normalized_diff(arr_st3[5], arr_st3[6])

ep.plot_bands(ndmi3, cmap="RdYlGn", cols=1, vmin=-1, vmax=1, figsize=(10, 14))

plt.show()


# In[15]:


len(ndmi3[ndmi3 > 0.4])


# # APR 2021

# In[16]:


landsat_bands4 = glob("Apr 2021////B?*.png")
landsat_bands4.sort()

p = []
for i in landsat_bands4:
    with rio.open(i, 'r') as f:
        p.append(f.read(1))

# Data
arr_st4 = np.stack(p)


# In[17]:


ep.plot_bands(arr_st4, cmap = 'gist_earth', figsize = (20, 12), cols = 6, cbar = False)
plt.show()


# In[18]:


ndmi4 = es.normalized_diff(arr_st4[5], arr_st4[6])

ep.plot_bands(ndmi4, cmap="RdYlGn", cols=1, vmin=-1, vmax=1, figsize=(10, 14))

plt.show()


# In[19]:


len(ndmi4[ndmi4 > 0.4])


# # May 2021

# In[20]:


landsat_bands5 = glob("May 2021//B?*.png")
landsat_bands5.sort()

q = []
for i in landsat_bands5:
    with rio.open(i, 'r') as f:
        q.append(f.read(1))

# Data
arr_st5 = np.stack(q)


# In[21]:


ep.plot_bands(arr_st5, cmap = 'gist_earth', figsize = (20, 12), cols = 6, cbar = False)
plt.show()


# In[22]:


ndmi5 = es.normalized_diff(arr_st5[5], arr_st5[6])

ep.plot_bands(ndmi5, cmap="RdYlGn", cols=1, vmin=-1, vmax=1, figsize=(10, 14))

plt.show()


# In[23]:


len(ndmi5[ndmi5 > 0.4])


# # June 2021

# In[24]:


landsat_bands6 = glob("June 2021//B?*.png")
landsat_bands6.sort()

r = []
for i in landsat_bands6:
    with rio.open(i, 'r') as f:
        r.append(f.read(1))

# Data
arr_st6 = np.stack(r)


# In[25]:


ep.plot_bands(arr_st6, cmap = 'gist_earth', figsize = (20, 12), cols = 6, cbar = False)
plt.show()


# In[26]:


ndmi6 = es.normalized_diff(arr_st6[5], arr_st6[6])

ep.plot_bands(ndmi6, cmap="RdYlGn", cols=1, vmin=-1, vmax=1, figsize=(10, 14))


# In[27]:


len(ndmi6[ndmi6 > 0.4])


# # July 2021

# In[28]:


landsat_bands7 = glob("July 2021//B?*.png")
landsat_bands7.sort()

s = []
for i in landsat_bands7:
    with rio.open(i, 'r') as f:
        s.append(f.read(1))

# Data
arr_st7 = np.stack(s)


# In[29]:


ep.plot_bands(arr_st7, cmap = 'gist_earth', figsize = (20, 12), cols = 6, cbar = False)
plt.show()


# In[30]:


ndmi7 = es.normalized_diff(arr_st7[5], arr_st7[6])

ep.plot_bands(ndmi7, cmap="RdYlGn", cols=1, vmin=-1, vmax=1, figsize=(10, 14))

plt.show()


# In[31]:


len(ndmi7[ndmi7 > 0.4])


# # August 2021

# In[32]:


landsat_bands8 = glob("Aug 2021///B?*.png")
landsat_bands8.sort()

t = []
for i in landsat_bands8:
    with rio.open(i, 'r') as f:
        t.append(f.read(1))

# Data
arr_st8 = np.stack(t)


# In[33]:


ep.plot_bands(arr_st8, cmap = 'gist_earth', figsize = (20, 12), cols = 6, cbar = False)
plt.show()


# In[34]:


ndmi8 = es.normalized_diff(arr_st8[5], arr_st8[6])

ep.plot_bands(ndmi8, cmap="RdYlGn", cols=1, vmin=-1, vmax=1, figsize=(10, 14))

plt.show()


# In[35]:


len(ndmi8[ndmi8 > 0.4])


# # September 2021

# In[36]:


landsat_bands9 = glob("SEP 2021//B?*.png")
landsat_bands9.sort()

u = []
for i in landsat_bands9:
    with rio.open(i, 'r') as f:
        u.append(f.read(1))

# Data
arr_st9 = np.stack(u)


# In[37]:


ndmi9 = es.normalized_diff(arr_st9[5], arr_st9[6])

ep.plot_bands(ndmi9, cmap="RdYlGn", cols=1, vmin=-1, vmax=1, figsize=(10, 14))

plt.show()


# In[38]:


len(ndmi9[ndmi9 > 0.4])


# # October 2020

# In[39]:


landsat_bands10 = glob("Oct 2020///B?*.png")
landsat_bands10.sort()

x = []
for i in landsat_bands10:
    with rio.open(i, 'r') as f:
        x.append(f.read(1))

# Data
arr_st10 = np.stack(x)


# In[40]:


ndmi10 = es.normalized_diff(arr_st10[5], arr_st10[6])

ep.plot_bands(ndmi10, cmap="RdYlGn", cols=1, vmin=-1, vmax=1, figsize=(10, 14))

plt.show()


# In[41]:


len(ndmi10[ndmi10 > 0.4])


# # November 2020

# In[42]:


landsat_bands11 = glob("Nov 2020//B?*.png")
landsat_bands11.sort()

v = []
for i in landsat_bands11:
    with rio.open(i, 'r') as f:
        v.append(f.read(1))

# Data
arr_st11 = np.stack(v)


# In[43]:


ndmi11 = es.normalized_diff(arr_st11[5], arr_st11[6])

ep.plot_bands(ndmi11, cmap="RdYlGn", cols=1, vmin=-1, vmax=1, figsize=(10, 14))

plt.show()


# In[44]:


len(ndmi11[ndmi11 > 0.4])


# # September 2020

# In[45]:


landsat_bands12 = glob("Sep 2020//B?*.png")
landsat_bands12.sort()

y = []
for i in landsat_bands12:
    with rio.open(i, 'r') as f:
        y.append(f.read(1))

# Data
arr_st12 = np.stack(y)


# In[46]:


ndmi12 = es.normalized_diff(arr_st12[5], arr_st12[6])

ep.plot_bands(ndmi12, cmap="RdYlGn", cols=1, vmin=-1, vmax=1, figsize=(10, 14))

plt.show()


# In[47]:


len(ndmi12[ndmi12 > 0.4])


# ## ARIMA MODEL

# In[48]:


import pmdarima as pm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA


# In[49]:


newdf = pd.read_csv ("./Sample1.csv")


# In[50]:


newdf.Value


# In[51]:


model = pm.auto_arima(newdf.Value, start_p=1, start_q=1,
                         test='adf',
                         max_p=3, max_q=3, m=12,
                         start_P=0, seasonal=True,
                         d=None, D=1, trace=True,
                         error_action='ignore',  
                         suppress_warnings=True, 
                         stepwise=True)


# In[52]:


model.plot_diagnostics(figsize=(10,8))
plt.show()


# In[53]:


n_periods = 10
future_months = ['Oct-21', 'Nov-21', 'Dec-21', 'Jan-22', 'Feb-22', 'Mar-22', 'Apr-22', 'May-22', 'Jun-22', 'July-22', 'Aug-22', 'Sep-22', 'Oct-22', 'Nov-22', 'Dec-22', 'Jan-23']
future_months = future_months[0:n_periods]

months = newdf.Month
months = months.append(pd.Series(future_months), ignore_index=True)
months 


# In[54]:



fc, confint = model.predict(n_periods=n_periods, return_conf_int=True)
index_of_fc = np.arange(len(newdf.Value), len(newdf.Value)+n_periods)
# make series for plotting purpose
fc_series = pd.Series(fc, index=index_of_fc)
lower_series = pd.Series(confint[:, 0], index=index_of_fc)
upper_series = pd.Series(confint[:, 1], index=index_of_fc)
# Plot
values = newdf.Value;
values = values.append(fc_series, ignore_index=True)
plt.plot(values[:32], label='Historical Data')
plt.plot(values[31:], color='darkgreen', label='Forecast Data')
plt.fill_between(lower_series.index, 
                 lower_series, 
                 upper_series, 
                 color='k', alpha=.15, label="Confidence interval")

#scale_factor = 1
#xmin, xmax = plt.xlim()

#plt.xlim(xmin * scale_factor, xmax * scale_factor)

# 
plt.xticks(range(0,len(months)),months, rotation=45)
N = 28

plt.gca().margins(x=0)
plt.gcf().canvas.draw()
tl = plt.gca().get_xticklabels()
maxsize = max([t.get_window_extent().width for t in tl])
m = 0.2 # inch margin
s = maxsize/plt.gcf().dpi*N+2*m
margin = m/plt.gcf().get_size_inches()[0]

plt.gcf().subplots_adjust(left=margin, right=1.-margin)
plt.gcf().set_size_inches(s, plt.gcf().get_size_inches()[1])
plt.grid()
plt.legend(loc='lower right')
plt.title("Water Stress Forecast")


# In[ ]:




