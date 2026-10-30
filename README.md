Electricity Consumption Prediction Using Linear Regression


PROJECT REPORT
Dataset: Individual Household Electric Power Consumption — UCI Machine Learning Repository
Tools: Python, Google Colab, Pandas, Matplotlib and Scikit-learn

1. Abstract

This project uses the real Individual Household Electric Power Consumption dataset from the UCI Machine Learning
Repository. The original dataset contains 2,075,259 one-minute measurements collected over almost four years. To
keep the project suitable for a beginner, only Date and Global_active_power are used. Missing values are removed,
minute-level readings are converted to daily averages, and a Linear Regression model is trained using the day number
as the input. The project produces a graph containing actual observations and the fitted regression line, and then uses
the model to estimate consumption for a future day.

2. Introduction
   
Electricity consumption changes over time because of household activities and appliance usage. Machine learning can
be used to study these changes and make predictions. Linear Regression is a simple supervised-learning algorithm
that predicts a continuous value by fitting a straight line to observed data. This makes it appropriate for demonstrating
the basic prediction workflow in a first-year project.

3. Dataset Description
   
The UCI dataset contains household electricity measurements from Sceaux, France, recorded between December
2006 and November 2010 at one-minute intervals. It has 2,075,259 instances and 9 features. The dataset contains
missing measurements in approximately 1.25% of rows


4. Problem Statement
   
Develop a small machine-learning project that reads the UCI household power dataset, cleans the required data,
creates a daily electricity-consumption series, plots the observations, fits a Linear Regression line, and predicts the
approximate consumption for a future day.

6. Project Methodology

Step 1 — Load the dataset
The UCI file is semicolon-separated and missing values are represented by '?'.
df = pd.read_csv('household_power_consumption.txt', sep=';', na_values='?')

Step 2 — Select columns
Only Date and Global_active_power are selected so that the project remains simple.
df = df[['Date', 'Global_active_power']]

Step 3 — Clean the data
The Date column is converted into date format. Global_active_power is converted to numeric values and rows with
missing power values are removed.
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df['Global_active_power'] = pd.to_numeric(df['Global_active_power'], errors='coerce')
df = df.dropna()

Step 4 — Create daily averages
The original file has one measurement per minute. Daily averaging reduces the data to one understandable value per
day.
daily = df.groupby('Date')['Global_active_power'].mean().reset_index()

8. Visualization
   
The daily series can be plotted with Date on the horizontal axis and average Global_active_power on the vertical axis.
The graph helps identify the overall movement of household power consumption.
plt.figure(figsize=(12,5))
plt.plot(daily['Date'], daily['Global_active_power'])
plt.xlabel('Date')
plt.ylabel('Average Electricity Consumption (kW)')
plt.title('Daily Electricity Consumption')
plt.show()

7. Linear Regression
   
For the simplified model, each day is represented by an integer: 0 for the first day, 1 for the second day, and so on. The
model uses this Day value to predict daily average Global_active_power.
Regression equation: n = bn + bnx
Here, x is the day number, n is the predicted electricity consumption, bn is the intercept, and bn is the slope learned
from the data.
daily['Day'] = range(len(daily))
X = daily[['Day']]
y = daily['Global_active_power']
model = LinearRegression()
model.fit(X, y)

9. Regression Graph
    
The regression graph contains the actual daily observations and the straight Linear Regression line. The points
represent measured data, while the line represents the general trend learned by the model.
plt.scatter(daily['Day'], daily['Global_active_power'], label='Actual Data')
predicted = model.predict(daily[['Day']])
plt.plot(daily['Day'], predicted, label='Linear Regression')
plt.xlabel('Day')
plt.ylabel('Average Electricity Consumption (kW)')
plt.title('Electricity Consumption - Linear Regression')
plt.legend()
plt.show()

10. Prediction
    
After training, the next day can be represented by the number of existing daily observations. The model then produces
an estimated electricity-consumption value.
next_day = [[len(daily)]]
prediction = model.predict(next_day)
print('Predicted consumption:', round(prediction[0], 2), 'kW')
