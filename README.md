# COGS 108 - Final Project 

# Notes
- add severity
- severity depends on hour, weather, visibility?
- visualization severity vs [weather, rush hour, hour, etc]
- fix intro (background, question, and hypothesis)
- hypothesis: how do certain factors affect the severity of the accident
- compare average severity of accident based on some factor


## Permissions

Place an `X` in the appropriate bracket below to specify if you would like your group's project to be made available to the public. (Note that PIDs will be scraped from the public submission, but student names will be included.)

* [  ] YES - make available
* [X] NO - keep private

# Overview

The car accidents rate in the US can be considered high.
We analyzed that some of the factors related to accidents are weather conditions, time of the incident, visibility, and severity. In this project, we will state our initial hypothesis about the factors causing accidents. With our large dataset, we will do cleaning, visualizing, and analyzing to come up with a conclusion to prove the correctness of our hypothesis.

# Names

- Carlos Wirawan
- Albert Estevan
- Nikolas Jody
- Nhat Tang
- Sheung Ho

# Group Members IDs

- A16112534
- A16093782
- A16105519
- A15565669
- A16081238

# Research Question

We are interested in what are the factors to cause traffic accidents in the US. As there are a lot more cars during rush hours, we want to find out the correlation between the period of time in a day/week (rush hour/non-rush hour, weekend/weekday) and the accident rate. Therefore, our research question is: Is there more accidents during rush hours due to the higher volume of traffic? At the same time, as the period of the day is not the only factor that causes a car accident, the other questions that we may also consider:
- Does the average severity directly correlate with another factor?
- Will the “rush-hour” time frame have the highest rate of accidents?
- Does “rush-hour” accidents have higher severity overall?
- Does the weather condition affect the accident rate?
- Does the visibility of the day affect the accident rate?

## Background and Prior Work

In the modern age, cars are everywhere and are the main form of transportation for many people in the U.S. As a result, driving has become an almost vital skill. With the sheer volume of cars on the road, there are bound to be many accidents that occur throughout the year in the country. In fact, there are an average of six million accidents a year happening in the U.S. alone and about three million people suffer injuries because of these accidents. With these numbers, it would seem that the roads are quite dangerous even with all the current regulations(Reference 1).

Thus, the goal of our project is to analyze the factors involved in accidents to see if there is some underlying connection between them. If there is a strong connection between accidents and a particular factor(s), it would be possible to reform the country based on the observed factor. For example, if the factor was time and it was observed that the time where most accidents tended to happen was rush hour, we can implement special regulations or change driver behavior so that drivers would drive more safely during this time.

Current studies reveal that there are many fatal accidents during evening rush hour as compared to morning rush hour, which may be a bit odd since there should be the same amount of cars on the road during either of these times. What is also interesting is that there tend to be more accidents during the weekends than during the weekdays. The statistics vary depending on state, but it is also important to consider that these statistics were only for 2016. Thus, it is unclear if accurate conclusions can be drawn from this data alone(Reference 2).
References:
- 1) https://www.driverknowledge.com/car-accident-statistics/
- 2) https://www.automoblog.net/2018/07/10/most-dangerous-times-to-drive/


# Hypothesis


We believe that the rate of car accidents will be higher during rush hour (3pm-6pm) on city streets. Because during rush hours people tend to be driving for a longer time than usual and the volume of cars will be high. On city streets, junctions are everywhere and that can contribute to higher number of accidents as junctions are prone to accidents.

Some factors of accidents are:
- Weather conditions : Certain weather conditions may cause  interference to a person's driving which may cause accidents.
- Traffic volume:  The higher the number of cars on the road may increase the chance of an accident occurring.
- Time of day: The time of day determines whether it is night or day in which the visibility differs and also rush hour time (3pm-6pm) where traffic volume increases significantly.
- Visibility: Having a low visibility will cause people to be vulnerable to accidents.
- Junctions: Accidents occured on junctions may have a high rate of severity.
- Severity: the rate of the accident’s impact to the traffic surrounding the accident.

# Dataset(s)

The data set that we are using is taken from the following web page.

Link: https://www.kaggle.com/sobhanmoosavi/us-accidents

Size: 1.05GB

Variable:
Time: The time of the accidents can be found in two of the columns of the dataset. One column indicates the estimated start time, and the other column indicates the estimated end time.
Location: The data set contains the estimated latitude and the longitude of the location where the accident occurs. We are planning to use these datas to present them on a map.
Number of accidents: The number of accidents will be the number of rows presented in the data set. This will be filtered by the start time of the accident.
Severity of accidents: The severity can be found in the column of the data set. It shows the severity of the accident, a number between 1 and 4, where 1 indicates the least impact on traffic.
Weather Condition: The weather is presented in the column of the data set, it indicates the weather when the accidents occured.

Accuracy of data: 
The data has more than a million accident reports that occured in the United States. We are certain that with this amount of datas, we can create a good analysis. The data is taken from  https://www.kaggle.com/, an online community of data scientists and machine learning practitioners. The kaggle user who posted the data cited that the data is taken from MapQuest, an American free online web mapping service owned by Verizon Media. Thus, we are certain that the source is legitimate.


# Setup


```python
# Import library

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import patsy 
import statsmodels.api as sm
from scipy.stats import ttest_ind, chisquare, normaltest

pd.set_option('precision', 2)
pd.options.display.max_rows = 7
pd.options.display.max_columns = 50
```


```python
file_path = 'US_Accidents_Dec19.csv'
df = pd.read_csv(file_path)
```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>Source</th>
      <th>TMC</th>
      <th>Severity</th>
      <th>Start_Time</th>
      <th>End_Time</th>
      <th>Start_Lat</th>
      <th>Start_Lng</th>
      <th>End_Lat</th>
      <th>End_Lng</th>
      <th>Distance(mi)</th>
      <th>Description</th>
      <th>Number</th>
      <th>Street</th>
      <th>Side</th>
      <th>City</th>
      <th>County</th>
      <th>State</th>
      <th>Zipcode</th>
      <th>Country</th>
      <th>Timezone</th>
      <th>Airport_Code</th>
      <th>Weather_Timestamp</th>
      <th>Temperature(F)</th>
      <th>Wind_Chill(F)</th>
      <th>Humidity(%)</th>
      <th>Pressure(in)</th>
      <th>Visibility(mi)</th>
      <th>Wind_Direction</th>
      <th>Wind_Speed(mph)</th>
      <th>Precipitation(in)</th>
      <th>Weather_Condition</th>
      <th>Amenity</th>
      <th>Bump</th>
      <th>Crossing</th>
      <th>Give_Way</th>
      <th>Junction</th>
      <th>No_Exit</th>
      <th>Railway</th>
      <th>Roundabout</th>
      <th>Station</th>
      <th>Stop</th>
      <th>Traffic_Calming</th>
      <th>Traffic_Signal</th>
      <th>Turning_Loop</th>
      <th>Sunrise_Sunset</th>
      <th>Civil_Twilight</th>
      <th>Nautical_Twilight</th>
      <th>Astronomical_Twilight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A-1</td>
      <td>MapQuest</td>
      <td>201.0</td>
      <td>3</td>
      <td>2016-02-08 05:46:00</td>
      <td>2016-02-08 11:00:00</td>
      <td>39.87</td>
      <td>-84.06</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.01</td>
      <td>Right lane blocked due to accident on I-70 Eas...</td>
      <td>NaN</td>
      <td>I-70 E</td>
      <td>R</td>
      <td>Dayton</td>
      <td>Montgomery</td>
      <td>OH</td>
      <td>45424</td>
      <td>US</td>
      <td>US/Eastern</td>
      <td>KFFO</td>
      <td>2016-02-08 05:58:00</td>
      <td>36.9</td>
      <td>NaN</td>
      <td>91.0</td>
      <td>29.68</td>
      <td>10.0</td>
      <td>Calm</td>
      <td>NaN</td>
      <td>0.02</td>
      <td>Light Rain</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>Night</td>
      <td>Night</td>
      <td>Night</td>
      <td>Night</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A-2</td>
      <td>MapQuest</td>
      <td>201.0</td>
      <td>2</td>
      <td>2016-02-08 06:07:59</td>
      <td>2016-02-08 06:37:59</td>
      <td>39.93</td>
      <td>-82.83</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.01</td>
      <td>Accident on Brice Rd at Tussing Rd. Expect del...</td>
      <td>2584.0</td>
      <td>Brice Rd</td>
      <td>L</td>
      <td>Reynoldsburg</td>
      <td>Franklin</td>
      <td>OH</td>
      <td>43068-3402</td>
      <td>US</td>
      <td>US/Eastern</td>
      <td>KCMH</td>
      <td>2016-02-08 05:51:00</td>
      <td>37.9</td>
      <td>NaN</td>
      <td>100.0</td>
      <td>29.65</td>
      <td>10.0</td>
      <td>Calm</td>
      <td>NaN</td>
      <td>0.00</td>
      <td>Light Rain</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>Night</td>
      <td>Night</td>
      <td>Night</td>
      <td>Day</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A-3</td>
      <td>MapQuest</td>
      <td>201.0</td>
      <td>2</td>
      <td>2016-02-08 06:49:27</td>
      <td>2016-02-08 07:19:27</td>
      <td>39.06</td>
      <td>-84.03</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.01</td>
      <td>Accident on OH-32 State Route 32 Westbound at ...</td>
      <td>NaN</td>
      <td>State Route 32</td>
      <td>R</td>
      <td>Williamsburg</td>
      <td>Clermont</td>
      <td>OH</td>
      <td>45176</td>
      <td>US</td>
      <td>US/Eastern</td>
      <td>KI69</td>
      <td>2016-02-08 06:56:00</td>
      <td>36.0</td>
      <td>33.3</td>
      <td>100.0</td>
      <td>29.67</td>
      <td>10.0</td>
      <td>SW</td>
      <td>3.5</td>
      <td>NaN</td>
      <td>Overcast</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>Night</td>
      <td>Night</td>
      <td>Day</td>
      <td>Day</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2974332</th>
      <td>A-2974356</td>
      <td>Bing</td>
      <td>NaN</td>
      <td>2</td>
      <td>2019-08-23 19:00:21</td>
      <td>2019-08-23 19:28:49</td>
      <td>33.78</td>
      <td>-117.85</td>
      <td>33.78</td>
      <td>-117.86</td>
      <td>0.56</td>
      <td>At Glassell St/Grand Ave - Accident. in the ri...</td>
      <td>NaN</td>
      <td>Garden Grove Fwy</td>
      <td>R</td>
      <td>Orange</td>
      <td>Orange</td>
      <td>CA</td>
      <td>92866</td>
      <td>US</td>
      <td>US/Pacific</td>
      <td>KSNA</td>
      <td>2019-08-23 18:53:00</td>
      <td>73.0</td>
      <td>73.0</td>
      <td>64.0</td>
      <td>29.74</td>
      <td>10.0</td>
      <td>SSW</td>
      <td>10.0</td>
      <td>0.00</td>
      <td>Partly Cloudy</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>Day</td>
      <td>Day</td>
      <td>Day</td>
      <td>Day</td>
    </tr>
    <tr>
      <th>2974333</th>
      <td>A-2974357</td>
      <td>Bing</td>
      <td>NaN</td>
      <td>2</td>
      <td>2019-08-23 19:00:21</td>
      <td>2019-08-23 19:29:42</td>
      <td>33.99</td>
      <td>-118.40</td>
      <td>33.98</td>
      <td>-118.40</td>
      <td>0.77</td>
      <td>At CA-90/Marina Fwy/Jefferson Blvd - Accident.</td>
      <td>NaN</td>
      <td>San Diego Fwy S</td>
      <td>R</td>
      <td>Culver City</td>
      <td>Los Angeles</td>
      <td>CA</td>
      <td>90230</td>
      <td>US</td>
      <td>US/Pacific</td>
      <td>KSMO</td>
      <td>2019-08-23 18:51:00</td>
      <td>71.0</td>
      <td>71.0</td>
      <td>81.0</td>
      <td>29.62</td>
      <td>10.0</td>
      <td>SW</td>
      <td>8.0</td>
      <td>0.00</td>
      <td>Fair</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>Day</td>
      <td>Day</td>
      <td>Day</td>
      <td>Day</td>
    </tr>
    <tr>
      <th>2974334</th>
      <td>A-2974358</td>
      <td>Bing</td>
      <td>NaN</td>
      <td>2</td>
      <td>2019-08-23 18:52:06</td>
      <td>2019-08-23 19:21:31</td>
      <td>34.13</td>
      <td>-117.23</td>
      <td>34.14</td>
      <td>-117.24</td>
      <td>0.54</td>
      <td>At Highland Ave/Arden Ave - Accident.</td>
      <td>NaN</td>
      <td>CA-210 W</td>
      <td>R</td>
      <td>Highland</td>
      <td>San Bernardino</td>
      <td>CA</td>
      <td>92346</td>
      <td>US</td>
      <td>US/Pacific</td>
      <td>KSBD</td>
      <td>2019-08-23 20:50:00</td>
      <td>79.0</td>
      <td>79.0</td>
      <td>47.0</td>
      <td>28.63</td>
      <td>7.0</td>
      <td>SW</td>
      <td>7.0</td>
      <td>0.00</td>
      <td>Fair</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>Day</td>
      <td>Day</td>
      <td>Day</td>
      <td>Day</td>
    </tr>
  </tbody>
</table>
<p>2974335 rows × 49 columns</p>
</div>



# Data Cleaning

* From the table above, we can see that there are a lot of columns, and most of them are going to be used. So we are going to only select used columns.



```python
# Only consider the columns used
df = df[['Start_Time','Weather_Condition','Visibility(mi)', 'Severity', 'Junction', 'Traffic_Signal']]
```

###### Drop NAN values


```python
# Drop NAN data
df = df.dropna(subset=['Visibility(mi)', 'Start_Time', 'Weather_Condition', 'Severity'])
```

Round up the decimal values.


```python
df['Visibility(mi)'] = df['Visibility(mi)'].round(0)
df['Visibility(mi)'].value_counts()
```




    10.0     2313768
    7.0        89504
    9.0        78885
              ...   
    63.0           1
    130.0          1
    67.0           1
    Name: Visibility(mi), Length: 46, dtype: int64



There are only a few accidents with visibility over 20, so we are going to change the ones above 20 to 20. And we log the count value for visualization purpose.


```python
def normalize(x):
    if x >20:
        return 20
    else:
        return x
    
def logNorm(x):
    return math.log10(x)
```


```python
df['Visibility(mi)'] = df['Visibility(mi)'].apply(normalize)
```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Start_Time</th>
      <th>Weather_Condition</th>
      <th>Visibility(mi)</th>
      <th>Severity</th>
      <th>Junction</th>
      <th>Traffic_Signal</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2016-02-08 05:46:00</td>
      <td>Light Rain</td>
      <td>10.0</td>
      <td>3</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2016-02-08 06:07:59</td>
      <td>Light Rain</td>
      <td>10.0</td>
      <td>2</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2016-02-08 06:49:27</td>
      <td>Overcast</td>
      <td>10.0</td>
      <td>2</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2974332</th>
      <td>2019-08-23 19:00:21</td>
      <td>Partly Cloudy</td>
      <td>10.0</td>
      <td>2</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2974333</th>
      <td>2019-08-23 19:00:21</td>
      <td>Fair</td>
      <td>10.0</td>
      <td>2</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2974334</th>
      <td>2019-08-23 18:52:06</td>
      <td>Fair</td>
      <td>7.0</td>
      <td>2</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>2901212 rows × 6 columns</p>
</div>




```python
visibility_counts = df['Visibility(mi)'].value_counts()
visibility_counts
```




    10.0    2313768
    7.0       89504
    9.0       78885
             ...   
    14.0          6
    19.0          2
    16.0          1
    Name: Visibility(mi), Length: 19, dtype: int64




```python
visibility_counts = visibility_counts.apply(logNorm)
```


```python
# Sort Index
visibility_counts = visibility_counts.sort_index()
```

Set the index to a list as the x axis, set the values as the y axis.


```python
x = visibility_counts.index.tolist()
y = visibility_counts.tolist()
plt.plot(x,y)
```




    [<matplotlib.lines.Line2D at 0x7f3577c0e160>]




![png](output_35_1.png)


###### Clean the start-time and end time

Clear the date


```python
df_updated = df.replace(to_replace ='[0-9]{4}-[0-9]{2}-[0-9]{2} ', value = '', regex = True)
```

Make a new column called 'Hour' to keep track of the time round down to the nearest hour.


```python
df_updated=df_updated.assign(Hour = df_updated['Start_Time'])
df_updated.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Start_Time</th>
      <th>Weather_Condition</th>
      <th>Visibility(mi)</th>
      <th>Severity</th>
      <th>Junction</th>
      <th>Traffic_Signal</th>
      <th>Hour</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>05:46:00</td>
      <td>Light Rain</td>
      <td>10.0</td>
      <td>3</td>
      <td>False</td>
      <td>False</td>
      <td>05:46:00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>06:07:59</td>
      <td>Light Rain</td>
      <td>10.0</td>
      <td>2</td>
      <td>False</td>
      <td>False</td>
      <td>06:07:59</td>
    </tr>
    <tr>
      <th>2</th>
      <td>06:49:27</td>
      <td>Overcast</td>
      <td>10.0</td>
      <td>2</td>
      <td>False</td>
      <td>True</td>
      <td>06:49:27</td>
    </tr>
    <tr>
      <th>3</th>
      <td>07:23:34</td>
      <td>Mostly Cloudy</td>
      <td>9.0</td>
      <td>3</td>
      <td>False</td>
      <td>False</td>
      <td>07:23:34</td>
    </tr>
    <tr>
      <th>4</th>
      <td>07:39:07</td>
      <td>Mostly Cloudy</td>
      <td>6.0</td>
      <td>2</td>
      <td>False</td>
      <td>True</td>
      <td>07:39:07</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_updated['Hour'] = df_updated['Hour'].replace(to_replace =':[0-9]{2}:[0-9]{2}', value = '', regex = True)
df_updated.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Start_Time</th>
      <th>Weather_Condition</th>
      <th>Visibility(mi)</th>
      <th>Severity</th>
      <th>Junction</th>
      <th>Traffic_Signal</th>
      <th>Hour</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>05:46:00</td>
      <td>Light Rain</td>
      <td>10.0</td>
      <td>3</td>
      <td>False</td>
      <td>False</td>
      <td>05</td>
    </tr>
    <tr>
      <th>1</th>
      <td>06:07:59</td>
      <td>Light Rain</td>
      <td>10.0</td>
      <td>2</td>
      <td>False</td>
      <td>False</td>
      <td>06</td>
    </tr>
    <tr>
      <th>2</th>
      <td>06:49:27</td>
      <td>Overcast</td>
      <td>10.0</td>
      <td>2</td>
      <td>False</td>
      <td>True</td>
      <td>06</td>
    </tr>
    <tr>
      <th>3</th>
      <td>07:23:34</td>
      <td>Mostly Cloudy</td>
      <td>9.0</td>
      <td>3</td>
      <td>False</td>
      <td>False</td>
      <td>07</td>
    </tr>
    <tr>
      <th>4</th>
      <td>07:39:07</td>
      <td>Mostly Cloudy</td>
      <td>6.0</td>
      <td>2</td>
      <td>False</td>
      <td>True</td>
      <td>07</td>
    </tr>
  </tbody>
</table>
</div>



Group them by rush hour and non rush hour


```python
df_hour=df_updated.groupby('Hour').size()

# change string to int
df_updated['Hour'] = pd.to_numeric(df_updated['Hour'])
```


```python
df_updated['Rush_Hour'] = np.where(((df_updated['Hour']>=16) & (df_updated['Hour']<20) | (df_updated['Hour'] >= 6) & (df_updated['Hour'] < 10)), True, False)
df_updated
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Start_Time</th>
      <th>Weather_Condition</th>
      <th>Visibility(mi)</th>
      <th>Severity</th>
      <th>Junction</th>
      <th>Traffic_Signal</th>
      <th>Hour</th>
      <th>Rush_Hour</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>05:46:00</td>
      <td>Light Rain</td>
      <td>10.0</td>
      <td>3</td>
      <td>False</td>
      <td>False</td>
      <td>5</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>06:07:59</td>
      <td>Light Rain</td>
      <td>10.0</td>
      <td>2</td>
      <td>False</td>
      <td>False</td>
      <td>6</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>06:49:27</td>
      <td>Overcast</td>
      <td>10.0</td>
      <td>2</td>
      <td>False</td>
      <td>True</td>
      <td>6</td>
      <td>True</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2974332</th>
      <td>19:00:21</td>
      <td>Partly Cloudy</td>
      <td>10.0</td>
      <td>2</td>
      <td>True</td>
      <td>False</td>
      <td>19</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2974333</th>
      <td>19:00:21</td>
      <td>Fair</td>
      <td>10.0</td>
      <td>2</td>
      <td>False</td>
      <td>False</td>
      <td>19</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2974334</th>
      <td>18:52:06</td>
      <td>Fair</td>
      <td>7.0</td>
      <td>2</td>
      <td>False</td>
      <td>False</td>
      <td>18</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
<p>2901212 rows × 8 columns</p>
</div>



Split the data set into two dataframes that contains only the rush hour accident and only the non rush hour accident.


```python
df_rushhour = df_updated[df_updated['Rush_Hour'] == True]
df_rushhour
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Start_Time</th>
      <th>Weather_Condition</th>
      <th>Visibility(mi)</th>
      <th>Severity</th>
      <th>Junction</th>
      <th>Traffic_Signal</th>
      <th>Hour</th>
      <th>Rush_Hour</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>06:07:59</td>
      <td>Light Rain</td>
      <td>10.0</td>
      <td>2</td>
      <td>False</td>
      <td>False</td>
      <td>6</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>06:49:27</td>
      <td>Overcast</td>
      <td>10.0</td>
      <td>2</td>
      <td>False</td>
      <td>True</td>
      <td>6</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>07:23:34</td>
      <td>Mostly Cloudy</td>
      <td>9.0</td>
      <td>3</td>
      <td>False</td>
      <td>False</td>
      <td>7</td>
      <td>True</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2974332</th>
      <td>19:00:21</td>
      <td>Partly Cloudy</td>
      <td>10.0</td>
      <td>2</td>
      <td>True</td>
      <td>False</td>
      <td>19</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2974333</th>
      <td>19:00:21</td>
      <td>Fair</td>
      <td>10.0</td>
      <td>2</td>
      <td>False</td>
      <td>False</td>
      <td>19</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2974334</th>
      <td>18:52:06</td>
      <td>Fair</td>
      <td>7.0</td>
      <td>2</td>
      <td>False</td>
      <td>False</td>
      <td>18</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
<p>1582878 rows × 8 columns</p>
</div>




```python
df_rushhour.groupby('Hour').size()
```




    Hour
    6     164525
    7     267036
    8     277671
           ...  
    17    216721
    18    163489
    19    112686
    Length: 8, dtype: int64




```python
df_nonrushhour = df_updated[df_updated['Rush_Hour'] == False]

df_nonrushhour
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Start_Time</th>
      <th>Weather_Condition</th>
      <th>Visibility(mi)</th>
      <th>Severity</th>
      <th>Junction</th>
      <th>Traffic_Signal</th>
      <th>Hour</th>
      <th>Rush_Hour</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>05:46:00</td>
      <td>Light Rain</td>
      <td>10.0</td>
      <td>3</td>
      <td>False</td>
      <td>False</td>
      <td>5</td>
      <td>False</td>
    </tr>
    <tr>
      <th>20</th>
      <td>10:11:15</td>
      <td>Light Snow</td>
      <td>2.0</td>
      <td>2</td>
      <td>False</td>
      <td>False</td>
      <td>10</td>
      <td>False</td>
    </tr>
    <tr>
      <th>21</th>
      <td>10:24:27</td>
      <td>Mostly Cloudy</td>
      <td>10.0</td>
      <td>2</td>
      <td>False</td>
      <td>False</td>
      <td>10</td>
      <td>False</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2974314</th>
      <td>15:23:31</td>
      <td>Fair</td>
      <td>8.0</td>
      <td>2</td>
      <td>True</td>
      <td>False</td>
      <td>15</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2974315</th>
      <td>15:33:46</td>
      <td>Fair</td>
      <td>10.0</td>
      <td>2</td>
      <td>True</td>
      <td>False</td>
      <td>15</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2974317</th>
      <td>15:45:43</td>
      <td>Fair</td>
      <td>10.0</td>
      <td>2</td>
      <td>True</td>
      <td>False</td>
      <td>15</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>1318334 rows × 8 columns</p>
</div>




```python
# Check out the size of the df
print(df_nonrushhour.shape)
print(df_rushhour.shape)
```

    (1318334, 8)
    (1582878, 8)
    

Visualize and compare the visibility of accidents occured dduring rush hour and non rush hour.


```python
df_rushhour['Visibility(mi)'].value_counts().sort_index()
```




    0.0     19068
    1.0     22800
    2.0     41132
            ...  
    16.0        1
    19.0        1
    20.0     7213
    Name: Visibility(mi), Length: 19, dtype: int64




```python
df_nonrushhour['Visibility(mi)'].value_counts().sort_index()
```




    0.0     10296
    1.0     15705
    2.0     30670
            ...  
    15.0     1098
    19.0        1
    20.0     3654
    Name: Visibility(mi), Length: 18, dtype: int64



Clean Weather data and group them into 4 categories:
- Percipitation
- Clear
- Cloudy
- Other


```python
def standardize_weather(string):
    
    string = string.lower()
    string = string.strip()
    
    if 'snow' in string:
        output = 'PRECIPITATION'
    elif 'ice' in string:
        output = 'PRECIPITATION'
    elif 'wintry' in string:
        output = 'PRECIPITATION'
    elif 'hail' in string:
        output = 'PRECIPITATION'
    elif 'clear' in string:
        output = 'CLEAR'
    elif 'rain' in string:
        output = 'PRECIPITATION'
    elif 'drizzle' in string:
        output = 'PRECIPITATION'
    elif 'thunder' in string:
        output = 'PRECIPITATION'
    elif 't-storm' in string:
        output = 'PRECIPITATION'
    elif 'haze' in string:
        output = 'OTHER'
    elif 'mist' in string:
        output = 'OTHER'
    elif 'partly cloudy' in string:
        output = 'CLEAR'
    elif 'cloud' in  string:
        output = 'CLOUDY'
    elif 'overcast' in  string:
        output = 'CLOUDY'
    elif 'fair' in  string:
        output = 'CLEAR'
    else:
        output = 'OTHER'
    return output
```


```python
df_updated = df_updated.dropna(subset=['Weather_Condition'])
df_updated['Weather_Condition'] = df_updated['Weather_Condition'].apply(standardize_weather)
```


```python
df_updated.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Start_Time</th>
      <th>Weather_Condition</th>
      <th>Visibility(mi)</th>
      <th>Severity</th>
      <th>Junction</th>
      <th>Traffic_Signal</th>
      <th>Hour</th>
      <th>Rush_Hour</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>05:46:00</td>
      <td>PRECIPITATION</td>
      <td>10.0</td>
      <td>3</td>
      <td>False</td>
      <td>False</td>
      <td>5</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>06:07:59</td>
      <td>PRECIPITATION</td>
      <td>10.0</td>
      <td>2</td>
      <td>False</td>
      <td>False</td>
      <td>6</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>06:49:27</td>
      <td>CLOUDY</td>
      <td>10.0</td>
      <td>2</td>
      <td>False</td>
      <td>True</td>
      <td>6</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>07:23:34</td>
      <td>CLOUDY</td>
      <td>9.0</td>
      <td>3</td>
      <td>False</td>
      <td>False</td>
      <td>7</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>07:39:07</td>
      <td>CLOUDY</td>
      <td>6.0</td>
      <td>2</td>
      <td>False</td>
      <td>True</td>
      <td>7</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_updated['junct_traff'] = np.where(((df_updated['Junction']==True) | (df_updated['Traffic_Signal'] == True)), True, False)
df_updated
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Start_Time</th>
      <th>Weather_Condition</th>
      <th>Visibility(mi)</th>
      <th>Severity</th>
      <th>Junction</th>
      <th>Traffic_Signal</th>
      <th>Hour</th>
      <th>Rush_Hour</th>
      <th>junct_traff</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>05:46:00</td>
      <td>PRECIPITATION</td>
      <td>10.0</td>
      <td>3</td>
      <td>False</td>
      <td>False</td>
      <td>5</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>06:07:59</td>
      <td>PRECIPITATION</td>
      <td>10.0</td>
      <td>2</td>
      <td>False</td>
      <td>False</td>
      <td>6</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>06:49:27</td>
      <td>CLOUDY</td>
      <td>10.0</td>
      <td>2</td>
      <td>False</td>
      <td>True</td>
      <td>6</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2974332</th>
      <td>19:00:21</td>
      <td>CLEAR</td>
      <td>10.0</td>
      <td>2</td>
      <td>True</td>
      <td>False</td>
      <td>19</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2974333</th>
      <td>19:00:21</td>
      <td>CLEAR</td>
      <td>10.0</td>
      <td>2</td>
      <td>False</td>
      <td>False</td>
      <td>19</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2974334</th>
      <td>18:52:06</td>
      <td>CLEAR</td>
      <td>7.0</td>
      <td>2</td>
      <td>False</td>
      <td>False</td>
      <td>18</td>
      <td>True</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>2901212 rows × 9 columns</p>
</div>




```python
df_updated['junct_traff_rush'] = np.where((((df_updated['Junction']==True) | (df_updated['Traffic_Signal'] == True)) & (df_updated['Rush_Hour'] == True)), True, False)
df_updated
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Start_Time</th>
      <th>Weather_Condition</th>
      <th>Visibility(mi)</th>
      <th>Severity</th>
      <th>Junction</th>
      <th>Traffic_Signal</th>
      <th>Hour</th>
      <th>Rush_Hour</th>
      <th>junct_traff</th>
      <th>junct_traff_rush</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>05:46:00</td>
      <td>PRECIPITATION</td>
      <td>10.0</td>
      <td>3</td>
      <td>False</td>
      <td>False</td>
      <td>5</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>06:07:59</td>
      <td>PRECIPITATION</td>
      <td>10.0</td>
      <td>2</td>
      <td>False</td>
      <td>False</td>
      <td>6</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>06:49:27</td>
      <td>CLOUDY</td>
      <td>10.0</td>
      <td>2</td>
      <td>False</td>
      <td>True</td>
      <td>6</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2974332</th>
      <td>19:00:21</td>
      <td>CLEAR</td>
      <td>10.0</td>
      <td>2</td>
      <td>True</td>
      <td>False</td>
      <td>19</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2974333</th>
      <td>19:00:21</td>
      <td>CLEAR</td>
      <td>10.0</td>
      <td>2</td>
      <td>False</td>
      <td>False</td>
      <td>19</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2974334</th>
      <td>18:52:06</td>
      <td>CLEAR</td>
      <td>7.0</td>
      <td>2</td>
      <td>False</td>
      <td>False</td>
      <td>18</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>2901212 rows × 10 columns</p>
</div>




```python
df_updated['junct_traff_nonrush'] = np.where((((df_updated['Junction']==True) | (df_updated['Traffic_Signal'] == True)) & (df_updated['Rush_Hour'] == False)), True, False)
df_updated
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Start_Time</th>
      <th>Weather_Condition</th>
      <th>Visibility(mi)</th>
      <th>Severity</th>
      <th>Junction</th>
      <th>Traffic_Signal</th>
      <th>Hour</th>
      <th>Rush_Hour</th>
      <th>junct_traff</th>
      <th>junct_traff_rush</th>
      <th>junct_traff_nonrush</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>05:46:00</td>
      <td>PRECIPITATION</td>
      <td>10.0</td>
      <td>3</td>
      <td>False</td>
      <td>False</td>
      <td>5</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>06:07:59</td>
      <td>PRECIPITATION</td>
      <td>10.0</td>
      <td>2</td>
      <td>False</td>
      <td>False</td>
      <td>6</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>06:49:27</td>
      <td>CLOUDY</td>
      <td>10.0</td>
      <td>2</td>
      <td>False</td>
      <td>True</td>
      <td>6</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2974332</th>
      <td>19:00:21</td>
      <td>CLEAR</td>
      <td>10.0</td>
      <td>2</td>
      <td>True</td>
      <td>False</td>
      <td>19</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2974333</th>
      <td>19:00:21</td>
      <td>CLEAR</td>
      <td>10.0</td>
      <td>2</td>
      <td>False</td>
      <td>False</td>
      <td>19</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2974334</th>
      <td>18:52:06</td>
      <td>CLEAR</td>
      <td>7.0</td>
      <td>2</td>
      <td>False</td>
      <td>False</td>
      <td>18</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>2901212 rows × 11 columns</p>
</div>



# Data Analysis & Results

Let's take a look at what our data looks like when we visualize it using various graphs. First, let's look at how the number of car crashes varies by the hour of day.


```python
sns.countplot(x = 'Hour', data = df_updated)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f356ead0be0>




![png](output_62_1.png)



```python
# plt.rcParams['figure.figsize'] = (17, 7)
# sns.set()
# ax = sns.distplot((df_updated['Hour']), kde = False, bins = 12)
# print(ax.xaxis.set_ticks(np.arange(0, 23, 2)))
# del ax
```

    [<matplotlib.axis.XTick object at 0x7f35691b70b8>, <matplotlib.axis.XTick object at 0x7f356920a1d0>, <matplotlib.axis.XTick object at 0x7f356920acf8>, <matplotlib.axis.XTick object at 0x7f3569155d30>, <matplotlib.axis.XTick object at 0x7f3569171978>, <matplotlib.axis.XTick object at 0x7f3569171e48>, <matplotlib.axis.XTick object at 0x7f3569179358>, <matplotlib.axis.XTick object at 0x7f356915f0f0>, <matplotlib.axis.XTick object at 0x7f3569179a58>, <matplotlib.axis.XTick object at 0x7f3569179e48>, <matplotlib.axis.XTick object at 0x7f356917f400>, <matplotlib.axis.XTick object at 0x7f356917f978>]
    


![png](output_63_1.png)


Here, we see that the data distribution appears to be bimodal, spiking at around 8AM and 4PM. There must be something special about these times that correlates with a higher number of accidents. For one, these hours are within the morning and evening rush hours so that could be the factor that relates to a higher number of accidents. Let's try to figure out the probability that a car crash occurs during rush hour. Since we made a column in our dataframe earlier to sort each accident into rush hour and non-rush hour, we can just graph the data using a bar graph.


```python
sns.countplot(x = 'Rush_Hour', data = df_updated)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f35691196a0>




![png](output_65_1.png)


From this graph, we see that there are more accidents that occur during rush hour than during non-rush hours, but we need a value so we can do a more detailed analysis.


```python
probability_rush = sum(df_updated['Rush_Hour'])/len(df_updated['Rush_Hour'])
probability_rush
```




    0.5455919801793182



With this probability, we can predict how many accidents are rush-hour accidents from a random sample or group of accidents. If we repeat this procedure many times and graph the results, we will get something that looks like a binomial distribution. Let's say we have 10000 trials, so n = 10000 and we repeat the procedure 10000 times. For the sake of our analysis let's have the null hypothesis be that there is an equal chance that accidents occur during rush hour and during non-rush hour (probability = 0.5). If we make a binomial distribution using the probability from the null hypothesis and the probability that we got from our data, we can compare the two distributions to see if there is a significant difference between the two.


```python
#binomial distribution using probability from data
binom_rush = np.random.binomial(10000, probability_rush, 10000)
sns.distplot(binom_rush, kde = False, bins = 10)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f3569167e80>




![png](output_69_1.png)



```python
#binomial distribution using hypothesis probability
binom_hypo = np.random.binomial(10000, 0.5, 10000)
sns.distplot(binom_hypo, kde = False, bins = 10)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f35690e7ac8>




![png](output_70_1.png)



```python
#compare the distributions to see if there is a significant difference
t_val, p_val = ttest_ind(binom_rush, binom_hypo)
print(p_val)
```

    0.0
    


```python
#using an alpha level of 0.01
if p_val<0.01:
    print('There IS a significant difference between the two distributions.')
else:
    print('There is NOT a significant difference between the two distributions.')
```

    There IS a significant difference between the two distributions.
    

The data should tell us that there is a significant difference between the two distributions. From this, we can see that our data does not match with the null hypothesis (that the probability is 0.5) so we have enough evidence to reject it. Thus, we can conclude that the proability of car accidents being rush hour or non-rush hour is not 50%. From our evidence, we can say that it is more likely that car accidents occur during rush hour.

Next, let's visualize how the number of accidents varies with visibility.


```python
sns.distplot(df_updated['Visibility(mi)'], bins = 10)
```

    /opt/conda/lib/python3.6/site-packages/scipy/stats/stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    




    <matplotlib.axes._subplots.AxesSubplot at 0x7f35690aa940>




![png](output_75_2.png)


In this graph, we see that there is a large number of car accidents occuring when the visibility is around 10 miles. A visibility of 10 miles is pretty standard and common, but there may be some other factors contributing to this spike at 10 miles. How the data was recorded could also have played a role. Since a visibility of 10 miles is common, any small deviation from 10 could just be rounded up or down to 10.

Aside from visibility, we also have weather, so let's take a look at how the number of crashes changes with the weather.


```python
sns.countplot(x = 'Weather_Condition', data = df_updated)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f356900b6a0>




![png](output_78_1.png)


Here, we see many accidents happening when the weather is clear or just cloudy. So, we can assume most accidents happen on any regular day with decent weather. There doesn't seem to be much of a glaring relationship between the weather and the number of accidents. The high number of accidents in clear weather could simply be due to the fact that there are many clear days in the year and not much other types of weather. For other weather types such as weather carrying some sort of precipitation, we might assume that it is due the type of weather being less frequent in the year and thus having fewer accidents. During a storm, people might try to drive more safely, as the roads are more slippery and dangerous, decreasing the number of potential accidents. Others might stay home more during a storm, so there would be less cars out on the road, but there isn't information about the volume of cars in the dataset so we would have to do further research to reach a more definitive conclusion.


```python
# pd.plotting.scatter_matrix(df_updated[[ 'Visibility(mi)', 'Hour']])
```

While we're still discussing weather, let's see how the severity of accidents changes according to the weather condition.


```python
sns.countplot(x = 'Weather_Condition', hue = 'Severity', data = df_updated, order = ['PRECIPITATION', 'CLOUDY', 'CLEAR', 'OTHER'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f3568fb9f60>




![png](output_82_1.png)


Again, we see that clear weather has the highest number of accidents, which is consistent with what we saw with the previous graph. We also see that there is a pattern with each weather has many accidents that are rated with severity 2 and it starts to decrease as we move to severity 4. There are very few if any severity 1 accidents in each graph(we can't clearly see them on the graph, but we could go in the dataset and find them). Overall, there doesn't seem to be much of a correlation between the severity of the accident and the weather condition since the parttern is consistent across all weather types.

With the severity of the accident, let's see how the severity relates to whether or not the accident happened during rush hour.


```python
sns.countplot(x = 'Severity', hue='Rush_Hour', data = df_updated)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f3568ed6828>




![png](output_85_1.png)


From the graph, we can't really see many severity 1 accidents. With severity 2 and severity 3, there are more of these accidents during rush hour. However, there are more severity 4 accidents during non-rush hour. We can't just take the raw number of accidents for each category, though. Since there are more accidents during rush hour to begin with, we should take the ratio of


```python
sns.countplot(x = 'Severity', hue='junct_traff', data = df_updated)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f3569167e10>




![png](output_87_1.png)



```python
sns.countplot(x = 'Severity', hue='junct_traff_rush', data = df_updated)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f35661beb70>




![png](output_88_1.png)



```python
sns.countplot(x = 'Severity', hue='junct_traff_nonrush', data = df_updated)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f35650e4b38>




![png](output_89_1.png)



```python
def get_count(input):
    df_severe = df_updated[df_updated['Severity'] == input]
    series_severe = df_severe['Severity']
    df_severe_RH = df_severe[df_severe['Rush_Hour'] == True]
    df_severe_NRH = df_severe[df_severe['Rush_Hour'] == False]
    severe_NRH_count = df_severe_NRH['Severity'].count()
    severe_RH_count = df_severe_RH['Severity'].count()
    return (severe_NRH_count,severe_RH_count)
    
severe4_NRH_count, severe4_RH_count = get_count(4)
severe3_NRH_count, severe3_RH_count = get_count(3)
severe2_NRH_count, severe2_RH_count = get_count(2)
severe1_NRH_count, severe1_RH_count = get_count(1)

series_severity_RH = df_rushhour['Severity']
series_severity_NRH = df_nonrushhour['Severity']
severity_RH_count = series_severity_RH.count()
severity_NRH_count = series_severity_NRH.count()

NRH_ratio_severe1 = severe1_NRH_count / severity_NRH_count
RH_ratio_severe1 = severe1_RH_count / severity_RH_count 
NRH_ratio_severe2 = severe2_NRH_count / severity_NRH_count
RH_ratio_severe2 = severe2_RH_count / severity_RH_count 
NRH_ratio_severe3 = severe3_NRH_count / severity_NRH_count
RH_ratio_severe3 = severe3_RH_count / severity_RH_count 
NRH_ratio_severe4 = severe4_NRH_count / severity_NRH_count
RH_ratio_severe4 = severe4_RH_count / severity_RH_count 

RH_ratio = np.array([RH_ratio_severe1, RH_ratio_severe2, RH_ratio_severe3, RH_ratio_severe4]) 
NRH_ratio = np.array([NRH_ratio_severe1, NRH_ratio_severe2, NRH_ratio_severe3, NRH_ratio_severe4]) 
severity = np.array([1,2,3,4])
RH_ratio_ser = pd.Series(RH_ratio) 
NRH_ratio_ser = pd.Series(NRH_ratio) 
severity_ser = pd.Series(severity) 

frame = {'severity': severity_ser, 'Rush_Hour_ratio': RH_ratio_ser, 'Non_Rush_Hour_ratio': NRH_ratio_ser}

df_ratio_severity = pd.DataFrame(frame)
df_ratio_severity = df_ratio_severity.set_index('severity')
df_ratio_severity

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Rush_Hour_ratio</th>
      <th>Non_Rush_Hour_ratio</th>
    </tr>
    <tr>
      <th>severity</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>3.52e-04</td>
      <td>2.94e-04</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6.88e-01</td>
      <td>6.51e-01</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.89e-01</td>
      <td>3.07e-01</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.22e-02</td>
      <td>4.09e-02</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_updated.rename(columns={'Weather_Condition': 'Weather', 'Visibility(mi)': 'Visibility'}, inplace=True)
```


```python
print(df_updated)
```

            Start_Time        Weather  Visibility  Severity  Junction  \
    0         05:46:00  PRECIPITATION        10.0         3     False   
    1         06:07:59  PRECIPITATION        10.0         2     False   
    2         06:49:27         CLOUDY        10.0         2     False   
    ...            ...            ...         ...       ...       ...   
    2974332   19:00:21          CLEAR        10.0         2      True   
    2974333   19:00:21          CLEAR        10.0         2     False   
    2974334   18:52:06          CLEAR         7.0         2     False   
    
             Traffic_Signal  Hour  Rush_Hour  junct_traff  junct_traff_rush  \
    0                 False     5      False        False             False   
    1                 False     6       True        False             False   
    2                  True     6       True         True              True   
    ...                 ...   ...        ...          ...               ...   
    2974332           False    19       True         True              True   
    2974333           False    19       True        False             False   
    2974334           False    18       True        False             False   
    
             junct_traff_nonrush  
    0                      False  
    1                      False  
    2                      False  
    ...                      ...  
    2974332                False  
    2974333                False  
    2974334                False  
    
    [2901212 rows x 11 columns]
    


```python
outcome, predictors = patsy.dmatrices("Severity ~ Visibility", df_updated)
mod = sm.OLS(outcome, predictors)
res = mod.fit()
```


```python
print(res.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:               Severity   R-squared:                       0.000
    Model:                            OLS   Adj. R-squared:                  0.000
    Method:                 Least Squares   F-statistic:                     339.8
    Date:                Thu, 19 Mar 2020   Prob (F-statistic):           7.26e-76
    Time:                        16:18:30   Log-Likelihood:            -2.3307e+06
    No. Observations:             2901212   AIC:                         4.661e+06
    Df Residuals:                 2901210   BIC:                         4.661e+06
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Intercept      2.3807      0.001   1913.903      0.000       2.378       2.383
    Visibility    -0.0024      0.000    -18.433      0.000      -0.003      -0.002
    ==============================================================================
    Omnibus:                   435121.534   Durbin-Watson:                   1.461
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):           661295.720
    Skew:                           1.156   Prob(JB):                         0.00
    Kurtosis:                       3.347   Cond. No.                         37.3
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    

Since the coefficient of Visibility is close to zero, it doesn't affect anything. Hence, severity and visibility is not related linearly. The value of R-squared is approximately zero, meaning that the linear model that we are using does not fit the data very well. If we were to predict the severity, we would most likely get 2.3807 for any visibility distance, since the model is just; Severity = 2.3807 


```python
outcome, predictors = patsy.dmatrices("Severity ~ Hour", df_updated)
mod = sm.OLS(outcome, predictors)
res = mod.fit()
```


```python
print(res.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:               Severity   R-squared:                       0.001
    Model:                            OLS   Adj. R-squared:                  0.001
    Method:                 Least Squares   F-statistic:                     1776.
    Date:                Thu, 19 Mar 2020   Prob (F-statistic):               0.00
    Time:                        16:18:34   Log-Likelihood:            -2.3299e+06
    No. Observations:             2901212   AIC:                         4.660e+06
    Df Residuals:                 2901210   BIC:                         4.660e+06
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Intercept      2.3267      0.001   2843.644      0.000       2.325       2.328
    Hour           0.0026   6.23e-05     42.145      0.000       0.003       0.003
    ==============================================================================
    Omnibus:                   435986.925   Durbin-Watson:                   1.461
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):           662924.523
    Skew:                           1.157   Prob(JB):                         0.00
    Kurtosis:                       3.355   Cond. No.                         34.1
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    

Since the coefficient of hour is close to zero, hour of the day doesn't really have an impact. Severity and hour is not related linearly. The value of R-squared is 0.001, which means the linear model that we are using does not fit the data very well. if we were to predict the severity, we would most likely get 2.3267 for any hour of the day, since the model is just; Severity = 2.3267


```python
outcome, predictors = patsy.dmatrices("Severity ~ Weather", df_updated)
mod = sm.OLS(outcome, predictors)
res = mod.fit()
```


```python
print(res.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:               Severity   R-squared:                       0.002
    Model:                            OLS   Adj. R-squared:                  0.002
    Method:                 Least Squares   F-statistic:                     1554.
    Date:                Thu, 19 Mar 2020   Prob (F-statistic):               0.00
    Time:                        16:18:47   Log-Likelihood:            -2.3285e+06
    No. Observations:             2901212   AIC:                         4.657e+06
    Df Residuals:                 2901208   BIC:                         4.657e+06
    Df Model:                           3                                         
    Covariance Type:            nonrobust                                         
    ============================================================================================
                                   coef    std err          t      P>|t|      [0.025      0.975]
    --------------------------------------------------------------------------------------------
    Intercept                    2.3416      0.000   5204.615      0.000       2.341       2.342
    Weather[T.CLOUDY]            0.0274      0.001     40.225      0.000       0.026       0.029
    Weather[T.OTHER]            -0.0119      0.002     -5.620      0.000      -0.016      -0.008
    Weather[T.PRECIPITATION]     0.0696      0.001     62.115      0.000       0.067       0.072
    ==============================================================================
    Omnibus:                   435185.410   Durbin-Watson:                   1.463
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):           661208.226
    Skew:                           1.156   Prob(JB):                         0.00
    Kurtosis:                       3.355   Cond. No.                         7.34
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    

Since the coefficient of weather is close to zero, it doesn't really matter as much. Severity and Weather is not related linearly. The value of R-squared is 0.002, which means the linear model that we are using does not fit the data very well. If we were to predict the severity, we would most likely get 2.3416 for any Weather conditions, since the model is just; Severity = 2.3416 

# Ethics & Privacy

We have authority to access the dataset as it is a public dataset. There was no possible data bias after our examination as accident reports the data collected does not correlate with things such as stereotype perpetuation, confirmation bias, etc. We have decided on  using a dataset with a relatively low exposure of personally identifiable information (PII). The subjects in the dataset are anonymous because a unique identifier is used, there was also no unnecessary information collected from the subject. However, in our case, the subjects of the dataset are humans and we are not certain that they have been given information of consent nor did we provide a way for a subject to request their information to be removed from the dataset.

# Conclusion & Discussion

From our analysis we found that accidents are more likely to occur during rush hour than non-rush hour. In our hypothesis, we guessed that this would be the case since there would be a higher volume of cars out on the road and thus more accidents. While our findings do indicate that accidents are more likely to happen during rush hour, we cannot say for sure that the hour is the only factor that makes accidents more likely. There may be some other conditions during rush hour that affect the accident rate.

After using the Ordinary Least Squares Regression method, we found that hour, visibility and weather have little to no affect at predicting severity. we found that all of their coefficients for each conditions is close to zero. Hence it still depend heavily on volume, as the higher of volume of cars, it mostlikely has the same chance of getting an accident for everycar, but having more cars on the road increase the number of accidents.
    
When looking further into accidents in junctions and traffic lights, we assume that we would more likely have a  more accidents there. Because most four way intersection is usually where accidents happen. However our results also show that on junctions and traffic lights we only see a change in similar ratiosn but higher volumes for rush hours and lower volume for non rush hour. Hence our previous assumption holds, that higher volume of cars on the road increases accidents. 
    
In conclusion, we can assume that other factors only play a small part in accidents and volume is the main cause of accidents. We could have done further if we have the total values of cars on the road on a particular day and conditions. However the lack of data limits us for further exploration. 

# Team Contributions

*Specify who in your group worked on which parts of the project.*
