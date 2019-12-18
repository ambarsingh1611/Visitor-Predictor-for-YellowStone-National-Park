# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 12:17:33 2019

@author: Cosmic Dust
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os

os.chdir('C:/Users/CosmicDust/Documents/Statistics/Final Project')

df = pd.read_csv('ysnp.csv')

df.head()
cmap = sns.diverging_palette(10, 10, as_cmap=True)
plt.figure(figsize=(15,12)),sns.heatmap(df.corr(), annot=True,center=0,cmap='RdBu_r')

df.columns


# Extrating month and year from date column
df['Month'] = df['Year/Month/Day'].apply(lambda date: date.split('/')[1])
df['Year'] = df['Year/Month/Day'].apply(lambda date: date.split('/')[0])

#------------------

yty = df.groupby('Year')['Recreation Visits'].sum().pct_change()

yty[1:].to_csv('yty.csv')



plt.figure(figsize=(6,6)),sns.scatterplot(x='TotalSnowfall(In)',y='Recreation Visits',data=df)

df.isnull().sum()


# Divide the Data Based on open and close season
def open_close(data):
    if data in (['1','2','3','4','11','12']):
        return 'Close'
    else:
        return 'Open'

df['openClose'] = df['Month'].apply(open_close)


df['TotalSnowfall(In)'].mean()

# Divide the Data Based on accessiblity
def accessibility(data):
    if data <= 8:
        return 1
    else:
        return 0


df['accessibility'] = df['TotalSnowfall(In)'].apply(accessibility)




#---------------------------

np.where(df['3month Percent Change Airfare Costs'].isnull())[0]

sns.set_style('darkgrid')
plt.figure(figsize=(6,6)),sns.scatterplot(x='LowestTemperature(F)',y='Recreation Visits',data=df, hue='accessibility')
plt.figure(figsize=(6,6)),sns.scatterplot(x='HighestTemperature(F)',y='Recreation Visits',data=df, hue='accessibility')
plt.figure(figsize=(6,6)),sns.scatterplot(x='WarmestMinimumTemperature(F)',y='Recreation Visits',data=df, hue='openClose')
plt.figure(figsize=(6,6)),sns.scatterplot(x='ColdestMaximumTemperature(F)',y='Recreation Visits',data=df, hue='openClose')
plt.figure(figsize=(6,6)),sns.scatterplot(x='AverageMinimumTemperature(F)',y='Recreation Visits',data=df, hue='openClose')
plt.figure(figsize=(6,6)),sns.scatterplot(x='AverageMaximumTemperature(F)',y='Recreation Visits',data=df, hue='openClose')
plt.figure(figsize=(6,6)),sns.scatterplot(x='MeanTemperature(F)',y='Recreation Visits',data=df)
plt.figure(figsize=(6,6)),sns.scatterplot(x='TotalPrecipitation(In)',y='Recreation Visits',data=df)
plt.figure(figsize=(6,6)),sns.scatterplot(x='TotalSnowfall(In)',y='Recreation Visits',data=df,hue='accessibility')
plt.figure(figsize=(6,6)),sns.scatterplot(x='Max 24hrPrecipitation(In)',y='Recreation Visits',data=df)
plt.figure(figsize=(6,6)),sns.scatterplot(x='Max 24hrSnowfall(In)',y='Recreation Visits',data=df)
plt.figure(figsize=(6,6)),sns.scatterplot(x='3month Percent Change Airfare Costs',y='Recreation Visits',data=df,hue = 'openClose')
plt.figure(figsize=(6,6)),sns.scatterplot(x='3month Percent Change Food Away From Home Costs',y='Recreation Visits',data=df,hue = 'openClose')
plt.figure(figsize=(6,6)),sns.scatterplot(x='Consumer Price Index',y='Recreation Visits',data=df,hue = 'openClose')
plt.figure(figsize=(6,6)),sns.scatterplot(x='Unemployment Rate',y='Recreation Visits',data=df,hue = 'openClose')


plt.figure(figsize=(6,6)),sns.scatterplot(x='3mon',y='Recreation Visits',data=df)
plt.figure(figsize=(6,6)),sns.scatterplot(x='TotalSnowfall(In)',y='Recreation Visits',data=df, hue='openClose')

plt.figure(figsize=(10,10)),sns.boxenplot(x='Month', y='Recreation Visits', data=df,order=['1','2','3','4','5','6','7','8','9','10','11','12'])

plt.figure(figsize=(10,7)),sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='YlOrRd')

sns.barplot(x='Month',y='LowestTemperature(F)', data = df,order=['1','2','3','4','5','6','7','8','9','10','11','12'])
sns.barplot(x='Month',y='HighestTemperature(F)', data = df,order=['1','2','3','4','5','6','7','8','9','10','11','12'])
sns.barplot(x='Year/Month/Day',y='TotalSnowfall(In)', data = df,order=['1','2','3','4','5','6','7','8','9','10','11','12'])
sns.barplot(x='Month',y='3month Percent Change Airfare Costs', data = df,order=['1','2','3','4','5','6','7','8','9','10','11','12'])

df.columns



#------------------------------------------------------------------------------
#-----------------FOR 3 MONTH PERCENT CHANGE AIRFARE COSTS---------------------
#------------------------------------------------------------------------------

# Function to impute values in missing values in 3month Percent Change Airfare Costs

mean_3monthPercentChangeAirfareCosts = df.groupby('Month')['3month Percent Change Airfare Costs'].mean()
mean_3monthPercentChangeAirfareCosts

def impute_3MonthPercentChangeAirfareCosts(cols):
    ThreeMonthPercentChangeAirfareCosts = cols[0]
    Month = cols[1]
    
    if pd.isnull(ThreeMonthPercentChangeAirfareCosts):

        if Month == 1:
            return mean_3monthPercentChangeAirfareCosts[0]

        elif Month == 10:
            return mean_3monthPercentChangeAirfareCosts[1]
        
        elif Month == 11:
            return mean_3monthPercentChangeAirfareCosts[2]
    
        elif Month == 12:
            return mean_3monthPercentChangeAirfareCosts[3]
        
        elif Month == 2:
            return mean_3monthPercentChangeAirfareCosts[4]
        
        elif Month == 3:
            return mean_3monthPercentChangeAirfareCosts[5]
    
        elif Month == 4:
            return mean_3monthPercentChangeAirfareCosts[6]
        
        elif Month == 5:
            return mean_3monthPercentChangeAirfareCosts[7]
        
        elif Month == 6:
            return mean_3monthPercentChangeAirfareCosts[8]
    
        elif Month == 7:
            return mean_3monthPercentChangeAirfareCosts[9]
        
        elif Month == 8:
            return mean_3monthPercentChangeAirfareCosts[10]

        else:
            return mean_3monthPercentChangeAirfareCosts[11]

    else:
        return ThreeMonthPercentChangeAirfareCosts

# Impute the value

df['3month Percent Change Airfare Costs'] = df[['3month Percent Change Airfare Costs','Month']].apply(impute_3MonthPercentChangeAirfareCosts, axis=1)
df['3month Percent Change Airfare Costs'].isnull().sum()


#----------------------------For rest of the columns---------------------------


# Check indexes of the null values for each column
df[df['LowestTemperature(F)'].isnull()].index
df[df['HighestTemperature(F)'].isnull()].index
df[df['WarmestMinimumTemperature(F)'].isnull()].index
df[df['ColdestMaximumTemperature(F)'].isnull()].index
df[df['AverageMinimumTemperature(F)'].isnull()].index
df[df['AverageMaximumTemperature(F)'].isnull()].index
df[df['MeanTemperature(F)'].isnull()].index
df[df['TotalPrecipitation(In)'].isnull()].index
df[df['TotalSnowfall(In)'].isnull()].index
df[df['Max 24hrPrecipitation(In)'].isnull()].index
df[df['Max 24hrSnowfall(In)'].isnull()].index

# Check months on these index values
df['Month'].loc[[216, 278, 297, 348, 352]]

# Calculate mean of each column according to Month
mean_LowestTemperature = df.groupby('Month')['LowestTemperature(F)'].mean()
mean_HighestTemperature = df.groupby('Month')['HighestTemperature(F)'].mean()
mean_WarmestMinimumTemperature = df.groupby('Month')['WarmestMinimumTemperature(F)'].mean()
mean_ColdestMaximumTemperature = df.groupby('Month')['ColdestMaximumTemperature(F)'].mean()
mean_AverageMinimumTemperature = df.groupby('Month')['AverageMinimumTemperature(F)'].mean()
mean_AverageMaximumTemperature = df.groupby('Month')['AverageMaximumTemperature(F)'].mean()
mean_MeamTemperature = df.groupby('Month')['MeanTemperature(F)'].mean()
mean_TotalPrecipitation = df.groupby('Month')['TotalPrecipitation(In)'].mean()
mean_TotalSnowfall = df.groupby('Month')['TotalSnowfall(In)'].mean()
mean_Max24hrPrecipitation = df.groupby('Month')['Max 24hrPrecipitation(In)'].mean()
mean_Max24hrSnowfall = df.groupby('Month')['Max 24hrSnowfall(In)'].mean()



#------------------------------------------------------------------------------
#---------------------FOR LOWEST TEMPERATURE-----------------------------------
#------------------------------------------------------------------------------


# Function to impute values in missing values in LowestTemperature(F)

def impute_LowestTemp(cols):
    LowestTemperature = cols[0]
    Month = cols[1]
    
    if pd.isnull(LowestTemperature):

        if Month == 7:
            return mean_LowestTemperature[9]

        elif Month == 10:
            return mean_LowestTemperature[1]
        
        else:
           return mean_LowestTemperature[3]

    else:
        return LowestTemperature

# Impute the values
df['LowestTemperature(F)'] = df[['LowestTemperature(F)', 'Month']].apply(impute_LowestTemp, axis=1)

# Check if function worked?
df.isnull().sum()


#------------------------------------------------------------------------------
#------------------------FOR HIGHEST TEMPERATURE-------------------------------
#------------------------------------------------------------------------------

# Function to impute values in missing values in HighestTemperature(F)

def impute_HighestTemp(cols):
    HighestTemperature = cols[0]
    Month = cols[1]
    
    if pd.isnull(HighestTemperature):

        if Month == 7:
            return mean_HighestTemperature[9]

        elif Month == 10:
            return mean_HighestTemperature[1]
        
        else:
           return mean_HighestTemperature[3]

    else:
        return HighestTemperature

# Impute the values
df['HighestTemperature(F)'] = df[['HighestTemperature(F)', 'Month']].apply(impute_HighestTemp, axis=1)



#------------------------------------------------------------------------------
#-------------------FOR WARMEST MINIMUM TEMPERATURE----------------------------
#------------------------------------------------------------------------------

def impute_WarmestMinTemp(cols):
    WarmestMinimumTemperature = cols[0]
    Month = cols[1]

    if pd.isnull(WarmestMinimumTemperature):
        
        if Month == 7:
            return mean_WarmestMinimumTemperature[9]

        elif Month == 10:
            return mean_WarmestMinimumTemperature[1]
        
        else:
            return mean_WarmestMinimumTemperature[3]
      
    else:
        return WarmestMinimumTemperature


df['WarmestMinimumTemperature(F)'] = df[['WarmestMinimumTemperature(F)', 'Month']].apply(impute_WarmestMinTemp, axis=1)


#------------------------------------------------------------------------------
#-------------------FOR COLDEST MAXIMUM TEMPERATURE----------------------------
#------------------------------------------------------------------------------

def impute_ColdestMaxTemp(cols):
    ColdestMaximumTemperature = cols[0]
    Month = cols[1]
    
    if pd.isnull(ColdestMaximumTemperature):
        
        if Month == 7:
            return mean_ColdestMaximumTemperature[9]

        elif Month == 10:
            return mean_ColdestMaximumTemperature[1]
        
        else:
            return mean_ColdestMaximumTemperature[3]
      
    else:
        return ColdestMaximumTemperature    


df['ColdestMaximumTemperature(F)'] = df[['ColdestMaximumTemperature(F)', 'Month']].apply(impute_ColdestMaxTemp, axis=1)


#------------------------------------------------------------------------------
#-------------------FOR AVERAGE MINIMUM TEMPERATURE----------------------------
#------------------------------------------------------------------------------


def impute_AvgMinTemp(cols):
    AverageMinimumTemperature = cols[0]
    Month = cols[1]
    
    if pd.isnull(AverageMinimumTemperature):
        
        if Month == 7:
            return mean_AverageMinimumTemperature[9]

        elif Month == 10:
            return mean_AverageMinimumTemperature[1]
        
        else:
            return mean_AverageMinimumTemperature[3]
      
    else:
        return AverageMinimumTemperature


df['AverageMinimumTemperature(F)'] = df[['AverageMinimumTemperature(F)', 'Month']].apply(impute_AvgMinTemp, axis=1)


#------------------------------------------------------------------------------
#-------------------FOR AVERAGE MAXIMUM TEMPERATURE----------------------------
#------------------------------------------------------------------------------

def impute_AvgMaxTemp(cols):
    AverageMaximumTemperature = cols[0]
    Month = cols[1]
    
    if pd.isnull(AverageMaximumTemperature):
        
        if Month == 7:
            return mean_AverageMaximumTemperature[9]

        elif Month == 10:
            return mean_AverageMaximumTemperature[1]
        
        else:
            return mean_AverageMaximumTemperature[3]
      
    else:
        return AverageMaximumTemperature


df['AverageMaximumTemperature(F)'] = df[['AverageMaximumTemperature(F)', 'Month']].apply(impute_AvgMaxTemp, axis=1)


#------------------------------------------------------------------------------
#-------------------------FOR MEAN TEMPERATURE---------------------------------
#------------------------------------------------------------------------------

def impute_MeanTemp(cols):
    MeanTemperature = cols[0]
    Month = cols[1]
    
    if pd.isnull(MeanTemperature):
    
        if Month == 7:
            return mean_MeamTemperature[9]

        elif Month == 10:
            return mean_MeamTemperature[1]
        
        else:
            return mean_MeamTemperature[3]
      
    else:
        return MeanTemperature


df['MeanTemperature(F)'] = df[['MeanTemperature(F)', 'Month']].apply(impute_MeanTemp, axis=1)

#------------------------------------------------------------------------------
#-----------------------FOR TOTAL PERCIPITATION--------------------------------
#------------------------------------------------------------------------------

def impute_TotalPrec(cols):
    TotalPrecipitation = cols[0]
    Month = cols[1]
    
    if pd.isnull(TotalPrecipitation):
    
        if Month == 7:
            return mean_TotalPrecipitation[9]

        elif Month == 10:
            return mean_TotalPrecipitation[1]
        
        else:
            return mean_TotalPrecipitation[3]
      
    else:
        return TotalPrecipitation


df['TotalPrecipitation(In)'] = df[['TotalPrecipitation(In)', 'Month']].apply(impute_TotalPrec, axis=1)


#------------------------------------------------------------------------------
#--------------------------FOR TOTAL SNOWFALL----------------------------------
#------------------------------------------------------------------------------

def impute_TotalSnowfall(cols):
    TotalSnowfall = cols[0]
    Month = cols[1]
    
    if pd.isnull(TotalSnowfall):
    
        if Month == 7:
            return mean_TotalSnowfall[9]

        elif Month == 10:
            return mean_TotalSnowfall[1]
        
        elif Month == 9:
            return mean_Max24hrSnowfall[11]
        
        else:
            return mean_TotalSnowfall[3]
      
    else:
        return TotalSnowfall


df['TotalSnowfall(In)'] = df[['TotalSnowfall(In)', 'Month']].apply(impute_TotalSnowfall, axis=1)


#------------------------------------------------------------------------------
#----------------------FOR MAX 24 HR PERCIPITATION-----------------------------
#------------------------------------------------------------------------------

def impute_Max24hrPrec(cols):
    Max24hrPrecipitation = cols[0]
    Month = cols[1]
    
    if pd.isnull(Max24hrPrecipitation):
        
        if Month == 7:
            return mean_Max24hrPrecipitation[9]

        elif Month == 10:
            return mean_Max24hrPrecipitation[1]
        
        else:
            return mean_Max24hrPrecipitation[3]
      
    else:
        return Max24hrPrecipitation


df['Max 24hrPrecipitation(In)'] = df[['Max 24hrPrecipitation(In)', 'Month']].apply(impute_Max24hrPrec, axis=1)

#------------------------------------------------------------------------------
#-------------------------FOR MAX 24HR SNOWFALL--------------------------------
#------------------------------------------------------------------------------

def impute_Max24hrSnowfall(cols):
    Max24hrSnowfall = cols[0]
    Month = cols[1]
    
    if pd.isnull(Max24hrSnowfall):
        
        if Month == 7:
            return mean_Max24hrSnowfall[9]

        elif Month == 10:
            return mean_Max24hrSnowfall[1]
        
        elif Month == 9:
            return mean_Max24hrSnowfall[11]
        
        else:
            return mean_Max24hrSnowfall[3]
      
    else:
        return Max24hrSnowfall


df['Max 24hrSnowfall(In)'] = df[['Max 24hrSnowfall(In)', 'Month']].apply(impute_Max24hrSnowfall, axis=1)


df.isnull().sum()
df.columns



    
# Create new dataframe based on open and closed season
closeSeason = df[df['openClose']=='Close']
openSeason = df[df['openClose']=='Open']



# OPEN SEASON CLEAN DATASET
# Export to CSV for creating models in R
openSeason.to_csv('open.csv')




#---------------------------LOGISTIC REGRESSION--------------------------------

df.columns
X = df.drop(['Recreation Visits','Year/Month/Day',
       '3month Percent Change Airfare Costs',
       '3month Percent Change Food Away From Home Costs',
       '3month Percent Change Gasoline Costs',
       '3month Percent Change Jet Fuel Costs', 'Consumer Price Index',
       'Consumer Sentiment Index', 'Unemployment Rate', 'openClose', 'accessibility'],axis=1)
y = df['accessibility']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)



# Training the model

from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# Making Predictions

predictions = logmodel.predict(X_test)
y_test

# Checking metrics
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)