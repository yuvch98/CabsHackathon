import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
# # Define the file path, column names to remove keep, and number of rows to sample
# file_path = "venv/CABS.csv"
# use_cols =['tpep_pickup_datetime', 'tpep_dropoff_datetime']
# df = pd.read_csv(file_path, usecols=use_cols)
# df.to_csv('new_Cabs.csv', index=False)
#
# file_path = "new_Cabs.csv"
# df = pd.read_csv(file_path)
# # # copying the file to work on a safe file
# work_df = df.copy()
# print(f"shape = {work_df.shape}")
# print(f"Checking for nulls: {work_df.isnull().sum()}")
# # #adjusting the data
# work_df['tpep_pickup_datetime'] = pd.to_datetime(work_df['tpep_pickup_datetime'], format='%Y-%m-%d %H:%M:%S')
# # extract date and time into separate columns
# work_df['tpep_pickup_date'] = work_df['tpep_pickup_datetime'].dt.date
# work_df['hour_pickup'] = work_df['tpep_pickup_datetime'].dt.hour
# work_df.drop(['tpep_dropoff_datetime'],axis=1,inplace=True)
# table = pd.crosstab(index=[work_df['tpep_pickup_date'], work_df['hour_pickup']], columns=['count'])
# # # in order to work better on less time per refresh
# table.to_csv('final_cabs.csv', index=True)
# final_path = 'final_cabs.csv'
# final_df = pd.read_csv(final_path)
# final_df.set_index(['tpep_pickup_date','hour_pickup'],inplace=True)
# # #
# # # # NOTICE that the numbers represented are given from monday where monday is 0  to sunday where sunday is 6 in day_of_week
# # #
# final_df['day_of_week']= pd.to_datetime(final_df.index.get_level_values(0).astype(str))
# final_df['day_of_week'] = final_df['day_of_week'].dt.weekday
# final_df['month_bin'] = pd.to_datetime(final_df.index.get_level_values(0).astype(str))
# final_df['month_bin'] = final_df['month_bin'].dt.month
# # extract the hour component from the datetime format
# #define the bins and labels
# bins_days = [0, 6, 12, 18, 23]
# labels_days = [0, 1, 2, 3]
# bins_months = [1, 4, 8, 12]
# labels_months = [1, 0, 2]
# # bin the hours and assign labels to each bin
# final_df['pickup_bin'] = pd.cut(final_df.index.get_level_values(1), bins=bins_days, labels=labels_days, include_lowest=True)
# final_df['month_bin'] = pd.cut(pd.to_datetime((final_df.index.get_level_values(0))).month, bins=bins_months, labels=labels_months, include_lowest=True)
# final_df.to_csv('model_cabs.csv',index=True)
# building the model:
# since we do not have a lot of features, we will not do feature selection, and we wish to keep them.
# which means we will use RandomForestRegressor for linear regression
final_df = pd.read_csv("model_cabs.csv")
holidays=['12-31','03-17','07-04','10-31','12-24']
final_df.set_index(['tpep_pickup_date','hour_pickup'],inplace=True)
final_df['holiday']= [1 if date.strftime('%m-%d') in holidays else 0 for date in pd.to_datetime(final_df.index.get_level_values(0))]
print(final_df.columns)
def get_outliers(df, series):
  q1 = series.quantile(0.25)
  q3 = series.quantile(0.75)

  if q1*q3 == 0:
    iqr = abs(2*(q1+q3))
    toprange = iqr
    botrange = -toprange
  else:
    iqr = q3-q1
    toprange = q3 + iqr * 1.5
    botrange = q1 - iqr * 1.5

  outliers_top=df[series > toprange]
  outliers_bot= df[series < botrange]
  outliers = pd.concat([outliers_bot, outliers_top], axis=0)

  return (botrange, toprange, outliers)
botrange, toprange, outliers = get_outliers(final_df, final_df['count'])
print(toprange)
print(botrange)

print(outliers)

X = final_df.drop('count',axis=1)
y = final_df['count']
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12345)

# Create a Gradient Boosted Regression Trees model
model = RandomForestRegressor(n_estimators=1000, random_state=12345)

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

print(len(y_pred))
# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error:{rmse}")
features = ['day_of_week','month_bin' ,'pickup_bin' ,'holiday']

#bin the hours and assign labels to each bin

new_inputs = pd.read_csv(r'TestDataForStudents.csv')
bins_days = [0, 6, 12, 18, 23]
labels_days = [0, 1, 2, 3]
bins_months = [1, 4, 8, 12]
labels_months = [1, 0, 2]
checking_df = pd.DataFrame()
print(new_inputs.head(5))
date=[]
time=[]
for row in new_inputs.values.tolist():
  date.append(row[0])
  time.append(row[1])

print(time)
print(date)
checking_df['tpep_pickup_datetime'] = pd.to_datetime(new_inputs['Dates'])
checking_df['day_of_week'] = checking_df['tpep_pickup_datetime'].dt.dayofweek
checking_df['pickup_bin'] = pd.cut(time, bins=bins_days, labels=labels_days, include_lowest=True)
checking_df['month_bin'] = pd.cut(date, bins=bins_months, labels= labels_months, include_lowest=True)
checking_df['holiday'] = [1 if date.strftime('%m-%d') in holidays else 0 for date in pd.to_datetime(date)]
checking_df.set_index(checking_df['tpep_pickup_datetime'],inplace=True)

predictions = model.predict(checking_df[features])
print(predictions)

#uploading submissions
new_inputs['predictions']=predictions
new_inputs.to_csv('SubmissionTestDataForStudents.csv')
