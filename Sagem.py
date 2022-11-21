#!/usr/bin/env python
# coding: utf-8

# In[32]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as py


# In[33]:


accidents = pd.read_csv("cleaned_accidents.csv")
accidents.head()


# In[34]:


allday_lst=accidents.Start_Time.astype(str).str.split(' ')
allday_lst2=[item[0] for item in allday_lst]

print('For the 49 states in this dataset:')
print('There are {} total accidents.'.format(accidents.shape[0]))

print('There are {} unique days.'.format(len(set(allday_lst2))))
print('On average, there are {} accidents per day.'.format(round(accidents.shape[0]/len(set(allday_lst2)))))


# In[35]:


fig,ax=plt.subplots(1,2,figsize=(15,8))
#clr = ('lightgray','dimgray','black','snow','whitesmoke','gainsboro','silver','darkgray','grey')
accidents.City.value_counts().sort_values(ascending=False)[:10].sort_values().plot(kind='barh',ax=ax[0])
ax[0].set_title("Top 10 Accident Prone Cities",size=15)
ax[0].set_xlabel('City',size=18)


count=accidents['City'].value_counts()
groups=list(accidents['City'].value_counts().index)[:5]
counts=list(count[:5])
#counts.append(count.agg(sum)-count[:10].agg('sum'))
#groups.append('Other')
type_dict=pd.DataFrame({"group":groups,"counts":counts})
#clr1=('dimgray','gray','darkgray','lightgray','gainsboro','whitesmoke','linen','aliceblue','azure','linen','slategrey')
qx = type_dict.plot(kind='pie', y='counts', labels=groups,autopct='%1.1f%%', pctdistance=0.9, radius=1.2,ax=ax[1])
plt.legend(loc=0, bbox_to_anchor=(1.15,0.4)) 
plt.subplots_adjust(wspace =0.5, hspace =0)
plt.ioff()
plt.ylabel('')
pass


# In[36]:


plt.figure(figsize=(6,5))
chart = sns.countplot(x='Timezone', hue='Severity', data=accidents ,palette="mako_r")
plt.title("Count of Accidents by Timezone (resampled data)", size=15, y=1.05)
plt.show()


# In[37]:


accidents['Start_Time'] = pd.to_datetime(accidents['Start_Time'])


# In[38]:


accidents['End_Time'] = pd.to_datetime(accidents['End_Time'])


# In[39]:


accidents['Year'] = accidents['Start_Time'].dt.year
accidents['Month'] = accidents['Start_Time'].dt.month  # .dt.month_name()
accidents['Hour'] = accidents['Start_Time'].dt.hour

diff = accidents['End_Time'] - accidents['Start_Time']
accidents['DelayTime'] = round(diff.dt.seconds/3600,1)
year = accidents['Year'].value_counts()
month = accidents['Month'].value_counts().sort_index()
month_map = {1:'Jan' , 2:'Feb' , 3:'Mar' , 4:'Apr' , 5:'May' , 6:'Jun', 7:'Jul' , 8:'Aug' 
             , 9:'Sep',10:'Oct' , 11:'Nov' , 12:'Dec'}

hour_severity = accidents[['Hour' , 'Severity']].groupby('Hour').agg({'Hour' : 'count' , 'Severity' : 'mean'})

accidents['Day'] = accidents['Start_Time'].dt.dayofweek
day_severity = accidents[['Day' , 'Severity']].groupby('Day').agg({'Day' : 'count' , 'Severity' : 'mean'})
day_map = {0:'Monday' , 1:'Tuesday' , 2:'Wednesday' , 3:"Thursday" , 4:'Friday' , 5:"Saturday" , 6:'Sunday'}


# In[40]:


plt.figure(figsize=(10,5))
sns.countplot(x='Month', hue='Severity', data=accidents ,palette="magma_r")
plt.title('Count of Accidents by Month (resampled data)', size=15, y=1.05)
plt.show()


# In[41]:


accidents_years = pd.DataFrame(accidents['Year'].groupby(accidents['Year']).count())
accidents_years


# In[42]:


hour_severity = accidents[['Hour' , 'Severity']].groupby('Hour').agg({'Hour' : 'count' , 'Severity' : 'mean'})

accidents['Day'] = accidents['Start_Time'].dt.dayofweek
day_severity = accidents[['Day' , 'Severity']].groupby('Day').agg({'Day' : 'count' , 'Severity' : 'mean'})
day_map = {0:'Monday' , 1:'Tuesday' , 2:'Wednesday' , 3:"Thursday" , 4:'Friday' , 5:"Saturday" , 6:'Sunday'}


# In[43]:


fig, ax = plt.subplots(1,1,figsize = (14,6))

sns.set_context('paper')

f = sns.barplot(x=hour_severity['Hour'].index , y=hour_severity['Hour'], ax = ax, palette='Pastel2')
ax2 = ax.twinx()

ax2.plot(hour_severity['Severity'] , color='CornFlowerBlue', label='Severity',linewidth=3,
           linestyle='solid',marker='.',markersize=18, markerfacecolor='w',markeredgecolor='b',markeredgewidth='2')

sns.despine(left=True)

ax2.spines[('top')].set_visible(False)
ax2.spines[('right')].set_visible(False)
ax2.spines[('left')].set_visible(False)
ax.set_xlabel("Hours of the Day", fontdict = {'fontsize':12 , 'color':'MidnightBlue'} )
ax.set_ylabel("No. of Accidents")
ax2.set_ylabel("Severity of Accidents", rotation=270 ,labelpad=20)
ax.set_title('Accidents and Severity per Hour of the day', fontdict = {'fontsize':16 , 'color':'MidnightBlue'})
# ax.legend(loc=(0,1))
ax2.legend(loc=(0,0.8))

ax.annotate('Morning office rush' , xytext=(3,150000) , xy=(7,5000),arrowprops={'arrowstyle':'fancy' , 'color':'Red'})
ax.annotate('Office Returning rush' , xytext=(19,150000),xy=(16,5000),arrowprops={'arrowstyle':'fancy', 'color':'Red'})

fig.show()


# In[44]:


POI_features = ['Amenity','Bump','Crossing','Give_Way','Junction','No_Exit','Railway','Roundabout','Station','Stop','Traffic_Calming','Traffic_Signal']

fig, axs = plt.subplots(ncols=3, nrows=4, figsize=(15, 10))

plt.subplots_adjust(hspace=0.5,wspace = 0.5)
for i, feature in enumerate(POI_features, 1):    
    plt.subplot(3, 4, i)
    sns.countplot(x=feature, hue='Severity', data=accidents ,palette="viridis")
    
    plt.xlabel('{}'.format(feature), size=12, labelpad=3)
    plt.ylabel('Accident Count', size=12, labelpad=3)    
    plt.tick_params(axis='x', labelsize=12)
    plt.tick_params(axis='y', labelsize=12)
    
 
    plt.title('Count of Severity in {}'.format(feature), size=14, y=1.05)
fig.suptitle('Count of Accidents in POI Features (resampled data)',y=1.02, fontsize=16)
plt.show()


# In[45]:


POI_features = ['Crossing','Junction','Traffic_Signal']

fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(15, 10))

plt.subplots_adjust(hspace=0.5,wspace = 0.5)
for i, feature in enumerate(POI_features, 1):    
    plt.subplot(3, 3, i)
    sns.countplot(x=feature, hue='Severity', data=accidents ,palette="viridis")
    
    plt.xlabel('{}'.format(feature), size=12, labelpad=3)
    plt.ylabel('Accident Count', size=12, labelpad=3)    
    plt.tick_params(axis='x', labelsize=12)
    plt.tick_params(axis='y', labelsize=12)
    
 
    plt.title('Count of Severity in {}'.format(feature), size=14, y=1.05)
fig.suptitle('Count of Accidents in POI Features (resampled data)',y=1.02, fontsize=16)
plt.show()


# In[46]:


import plotly.express as px
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objects as go
import warnings


# In[47]:


from plotly.offline import init_notebook_mode,iplot,plot
init_notebook_mode(connected=True)


# In[48]:


accidents_sc = accidents[accidents['County'] == 'Santa Clara']

year = accidents_sc ['Year'].value_counts()
month = accidents_sc ['Month'].value_counts().sort_index()
month_map = {1:'Jan' , 2:'Feb' , 3:'Mar' , 4:'Apr' , 5:'May' , 6:'Jun', 7:'Jul' , 8:'Aug' 
             , 9:'Sep',10:'Oct' , 11:'Nov' , 12:'Dec'}
hour = accidents_sc ['Hour'].value_counts().sort_index()

hour_severity = accidents_sc [['Hour' , 'Severity']].groupby('Hour').agg({'Hour' : 'count' , 'Severity' : 'mean'})


day_severity = accidents_sc ['Day'].value_counts().sort_index()
day_map = {0:'Monday' , 1:'Tuesday' , 2:'Wednesday' , 3:"Thursday" , 4:'Friday' , 5:"Saturday" , 6:'Sunday'}
year_map = {x:x for x in year.index}
hour_map = {x:x for x in hour.index}

light_palette = sns.color_palette(palette='pastel')


# In[49]:


day_severity = accidents_sc['Day'].value_counts().sort_index()


# In[50]:


fig = px.density_mapbox(accidents_sc , lat='Start_Lat', lon='Start_Lng', z='Severity', radius=5,
                        center=dict(lat=37.3541079, lon=-121.9552356), zoom=12,
                        mapbox_style="open-street-map", height=900)
fig.show()


# In[51]:


accidents_sj = accidents[accidents['State'] == 'CA']


year = accidents_sj ['Year'].value_counts()
month = accidents_sj ['Month'].value_counts().sort_index()
month_map = {1:'Jan' , 2:'Feb' , 3:'Mar' , 4:'Apr' , 5:'May' , 6:'Jun', 7:'Jul' , 8:'Aug' 
             , 9:'Sep',10:'Oct' , 11:'Nov' , 12:'Dec'}
hour = accidents_sj ['Hour'].value_counts().sort_index()

hour_severity = accidents_sj [['Hour' , 'Severity']].groupby('Hour').agg({'Hour' : 'count' , 'Severity' : 'mean'})


day_severity = accidents_sj ['Day'].value_counts().sort_index()
day_map = {0:'Monday' ,  1:'Tuesday' , 2:'Wednesday' , 3:"Thursday" , 4:'Friday' , 5:"Saturday" , 6:'Sunday'}
year_map = {x:x for x in year.index}
hour_map = {x:x for x in hour.index}

light_palette = sns.color_palette(palette='pastel')


# In[52]:


day_severity = accidents_sj['Day'].value_counts().sort_index()


# In[53]:


fig = px.density_mapbox(accidents_sj , lat='Start_Lat', lon='Start_Lng', z='Severity', radius=5,
                        center=dict(lat=37.3541079, lon=-121.9552356), zoom=12,
                        mapbox_style="open-street-map", height=900)
fig.show()


# In[54]:


sv1= accidents[accidents['Severity']==1]
sv2 = accidents[accidents['Severity']==2]
sv3 = accidents[accidents['Severity']==3]
sv4 = accidents[accidents['Severity']==4]
plt.figure(figsize=(15,10))

plt.plot( 'Start_Lng', 'Start_Lat', data=sv1, linestyle='', marker='o', markersize=1.5, color="green", alpha=0.2, label='severity 1')

plt.plot( 'Start_Lng', 'Start_Lat', data=sv2, linestyle='', marker='o', markersize=1.5, color="blue", alpha=0.2, label='severity 2')

plt.plot( 'Start_Lng', 'Start_Lat', data=sv3, linestyle='', marker='o', markersize=1.5, color="dodgerblue", alpha=0.2, label='severity 3')

plt.plot( 'Start_Lng', 'Start_Lat', data=sv4, linestyle='', marker='o', markersize=1.5, color="red", alpha=0.2, label='severity 4')
plt.legend(markerscale=8)
plt.xlabel('Longitude', size=12, labelpad=3)
plt.ylabel('Latitude', size=12, labelpad=3)
plt.title('Map of Accidents', size=16, y=1.05)
plt.show()


# In[ ]:





# In[ ]:


fig = px.density_mapbox(accidents_sj , lat='Start_Lat', lon='Start_Lng', z='Severity', radius=5,
                        center=dict(lat=37.3541079, lon=-121.9552356), zoom=12,
                        mapbox_style="open-street-map", height=900)
fig.show()


# In[ ]:



# The location of accidents for each state
# Where are the accidents?
feature='Accident location'

# Set the state as the index
accidents.set_index('State',drop=True,inplace=True)

# State is the index when selecting bool type data as df_bool
acc_bool=accidents.select_dtypes(include=['bool'])

# Reset the index of the original data for other calculations
accidents.reset_index(inplace=True)

# Set the size of the figure
fig= plt.figure(figsize=(15,6))

# Cutoff percentage for display
pct_cutoff=2.5

# Define autopct: only display the value if the percentage is greater than the predefined cutoff value
def my_autopct(pct):
    return ('%1.0f%%' % pct) if pct > pct_cutoff else ''


# Run a for loop for each state
for i,state in enumerate(state_lst):
    
    # Set a sub plot
    plt.subplot(1, 3, 1+i)
    # Slice the dataframe for the specific state and feature
    acc_temp=acc_bool[acc_bool.index==state]
    acc_temp=(acc_temp.sum(axis=0)/acc_temp.sum(axis=0).sum()).sort_values()

    
    # Define lables to go with the pie plot
    labels = [n if v > pct_cutoff/100 else ''
              for n, v in zip(df_temp.index, acc_temp)] 
    
    # Generate the pie plot
    plt.pie(df_temp, labels=labels, autopct=my_autopct, shadow=True)
    
    # Set axis,label and title
    plt.axis('equal')
    plt.xlabel(feature)
    plt.title(state)

plt.xlabel(feature)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


fig = px.density_mapbox(accidents, lat='Start_Lat', lon='Start_Lng', z='Severity', radius=5,
                        center=dict(lat=40.730610, lon=-73.935242), zoom=12,
                        mapbox_style="open-street-map", height=900)
fig.show()


# In[ ]:




