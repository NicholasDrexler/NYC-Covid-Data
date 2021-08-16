'''
Recent Covid Visualizations for NYC
Created by: Nicholas Drexler 
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
day_span = 90                               # choose how long the time series should be

# import data that is constantly updated from the web using pandas
url = "https://raw.githubusercontent.com/nychealth/coronavirus-data/master/latest/now-data-by-day.csv"
data = pd.read_csv(url)
dates = data.date_of_interest[-day_span:]
cases = data.CASE_COUNT[-day_span:]
hosps = data.HOSPITALIZED_COUNT[-day_span:]
deaths = data.DEATH_COUNT[-day_span:]

N = np.arange(day_span) 
days = dates[0::(int(day_span/20))]         # controls how many dates are shown on x axis
dates_length = len(data.date_of_interest)
first_day = dates[dates_length-day_span]    # to display first day of study in title

# for predictive curves
predict_span = 10                           # how many days ahead to predict
dates_array = dates.to_numpy()
days_array = days.to_numpy()
case_array = cases.to_numpy()
n_predict = np.arange(len(N)-1,len(N)+predict_span-1)

# Certain maths
last_day = dates_array[-1]
twoweek_span = case_array[-15:]
tws_max = max(twoweek_span)
tws_avg = np.mean(twoweek_span)

# Controls Polynomial-fit degree 
poly_degree = 2                             # Used to change polynomial degree for regressions
caseD,hospD,deathD = 0,0,0                  # for 3rd degree poly fitting

# Quadratic Regression for Cases
case_curve = np.polyfit(N,cases,poly_degree)
case_poly = np.poly1d(case_curve)
caseA = str(round(case_curve[0],3))
caseB = str(round(case_curve[1],3))
caseC = str(round(case_curve[2],3))

# Quadratic Regression for Hospitalizations
hosp_curve = np.polyfit(N,hosps,poly_degree)
hosp_poly = np.poly1d(hosp_curve)
hospA = str(round(hosp_curve[0],3))
hospB = str(round(hosp_curve[1],3))
hospC = str(round(hosp_curve[2],3))

# Quadratic Regression for Deaths
death_curve = np.polyfit(N,deaths,poly_degree)
death_poly = np.poly1d(death_curve)
deathA = str(round(death_curve[0],3))
deathB = str(round(death_curve[1],3))
deathC = str(round(death_curve[2],3))

# handles sign + or - in the legend for each regression
sign = lambda x: '+ ' + str(float(x)) if float(x) >= 0 else '- ' + str(abs(float(x)))
if poly_degree == 2:
    case_equation = f"Case Regression: ${caseA}x^2 {sign(caseB)}x {sign(caseC)}$"
    hosp_equation = f"Hspitalization Regression: ${hospA}x^2 {sign(hospB)}x {sign(hospC)}$"
    death_equation = f"Death Regression: ${deathA}x^2 {sign(deathB)}x {sign(deathC)}$"
    
if poly_degree == 3:
    caseD = str(round(case_curve[3],3))
    hospD = str(round(case_curve[3],3))
    deathD = str(round(case_curve[3],3))
    case_equation = f"Case Regression: ${caseA}x^3 {sign(caseB)}x^2 {sign(caseC)}x {sign(caseD)}$"
    hosp_equation = f"Case Regression: ${hospA}x^3 {sign(hospB)}x^2 {sign(hospC)}x {sign(hospD)}$"
    death_equation = f"Case Regression: ${deathA}x^3 {sign(deathB)}x^2 {sign(deathC)}x {sign(deathD)}$"
 
################ Graphing ################
plt.suptitle("NYC Covid-19 Data",x=1,size=20)
plt.subplots_adjust(left=0,right=2,top=0.8,bottom=-2.5, hspace=0.25)

# Case plots
plt.subplot(3,1,1)
plt.title("NYC Covid-19 Cases as of {}".format(first_day),size=12)
plt.plot(dates,cases,"black",linewidth=0.8, label="Cases")
plt.plot(dates,case_poly(N),"green",linewidth=1, label=case_equation)
plt.plot(n_predict,case_poly(n_predict),"blue",linewidth=1, label="Cases Forcast")
plt.xticks(ticks=days, labels="")
plt.xlim(0.0, (day_span + predict_span))
plt.grid()
plt.legend(loc='upper center',bbox_to_anchor=(0.5, 0),prop={'size': 12},ncol=3)
plt.axvline(x = len(dates)-14, color = 'r', linewidth=2, label = '14 days from last data point')

# Hospitalization plots
plt.subplot(3,1,2)
plt.title("NYC Covid-19 Hospitalizations as of {}".format(first_day),size=12)
plt.plot(dates,hosps,"purple",linewidth=0.8, label="Hospitalizations")
plt.plot(dates,hosp_poly(N),"green",linewidth=0.8, label=hosp_equation)
plt.plot(n_predict,hosp_poly(n_predict),"blue",linewidth=0.8, label="Hospitalizations Forcast")
plt.xticks(ticks=days, labels="")
plt.xlim(0.0, (day_span + predict_span))
plt.grid()
plt.legend(loc='upper center',bbox_to_anchor=(0.5, 0),prop={'size': 12},ncol=3)
plt.axvline(x = len(dates)-14, color = 'r', linewidth=2, label = '14 days from last data point')

# Death plots
plt.subplot(3,1,3)
plt.title("NYC Covid-19 Deaths as of {}".format(first_day),size=12)
plt.plot(dates,deaths,"red",linewidth=0.8, label ="Deaths")
plt.plot(dates,death_poly(N),"green",linewidth=0.8, label=death_equation)
plt.plot(n_predict,death_poly(n_predict),"blue",linewidth=0.8, label="Deaths Forcast")
plt.xticks(rotation=90, ticks=days)
plt.xlim(0.0, (day_span + predict_span))
plt.grid()
plt.legend(loc='upper center',bbox_to_anchor=(0.5, -0.3),prop={'size': 12},ncol=3)
plt.axvline(x = len(dates)-14, color = 'r', linewidth=2, label = '14 days from last data point')

