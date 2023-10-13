from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from autogluon.timeseries.splitter import MultiWindowSplitter


df1 = pd.read_pickle('./data/final_dataset.pkl')
#df_filterd=unfiltered_df.groupby('cusip').filter(lambda x:(x.age.max() > 4))
#df_filterd.to_pickle('./data/bonds_filtered.pkl')
#df1 = pd.read_pickle('./data/bonds_filtered.pkl')

"""First Use case only bond and macro features"""
df1['dates'] = pd.to_datetime(df1['dates'].astype(str), format='%Y%m')
bond_macro_features = ['age','coupon','ratings','yields','duration','excreturn','one_month_lag','two_month_lag'
'three_month_lag','spread','6_month_momentum','volatility','skew','VaR' , 'mom6mspread', 'mom6xrtg', 'mom6industry', 'face_value',
'time_to_maturity', 't_bill', 'term_spread', 'inflation']

bond_features_d_t = ['cusip']+['dates']  + bond_macro_features
df = df1[df1.columns.intersection(bond_features_d_t)]
# Sort the data by date in ascending order within each panel (CUSIP ID)
drop_cusip = ['98978VAB9', '98978VAH6'] ##columsn have null values
df = df[~df['cusip'].isin(drop_cusip)]
data = df.sort_values(by=['cusip', 'dates'])

# Group the data by CUSIP ID
grouped = data.groupby('cusip')

# Create empty DataFrames to store train and test sets
train_set = pd.DataFrame(columns=data.columns)
test_set = pd.DataFrame(columns=data.columns)

# Iterate through each group (CUSIP ID) and Check if the time series length is at least 4 years so that there is enough
## length to do the validation, window and test. However, it can be done directly with Autogluon but we did it here by ourself because autogluon sometimes
## do not cut test data properly and since time series it has to be cut properly
for cusip, group_data in grouped:
    
    if len(group_data) >= 12 * 4:  # Assuming monthly data
        # Get the last 10 months' data for the test set
        test_data = group_data.tail(12)
        test_set = test_set.append(test_data)
        
        # Get all data for the train set because autogluon you specify prediction length which later becomes 
        ## outof sample data so it can be match with test data
        ##train_data = group_data.drop(test_data.index)
        train_set = train_set.append(group_data)

""" Please note that this is single window. However, we did multiple windows with rolling and fix and this can be done from the below code
 just uncomment if you want to do it also in fit you need to add extra param called num_val_windows. However, wit windows
 run time increase exponentially from 1 day to 3 days"""
#splitter = MultiWindowSplitter(num_windows=5)
#train_data, test_data_multi_window = splitter.split(test_data, prediction_length)
#
#predictor.evaluate(test_data_multi_window)
# Display information about the train and test sets
print("Train Set:")
print(train_set.head())
print("\nTest Set:")
print(test_set.head())
## Similar to tb_model autogluon does it it for you but you have to know the params or uncomment the below 2 lines
#train_set.interpolate(method='bfill', inplace=True)
#test_set.interpolate(method='bfill', inplace=True)
#df['dates'] = pd.to_datetime(df['dates'])
#
#df.interpolate(method='bfill', inplace=True)
train_set['dates'] = pd.to_datetime(train_set['dates'])
test_set['dates'] = pd.to_datetime(test_set['dates'])
prediction_length = 12 ### can be changed it is monthing but the months you want to predict
# Convert 'Time' column to datetime format
#df['dates'] = pd.to_datetime(df['dates'])
## this is time series format for Autogluon to do the modelling where you can say static feature like id in our case
## it is cusip
train_time_series = TimeSeriesDataFrame.from_data_frame(
    train_set,
    id_column="cusip",
    timestamp_column="dates"
)
print(train_time_series.freq)

## the steps we did before can be done with this one single line below but has to be done carefully
## and similarly to tb validation is done automatically by autogluon just speciy the years
#train_data, test_data = train_time_series.train_test_split(prediction_length)
## In the predictor you can specify known covariates i.e. the values you know in future where the past covariates 
## you do not have to specify because they are the variables picked by you
predictor = TimeSeriesPredictor(prediction_length=prediction_length,
    target="excreturn",
    eval_metric=  "RMSE",
    verbosity=4,
    ignore_time_index=True)
predictor.fit(train_time_series,   hyperparameters={
        "DirectTabularModel": {'tabular_hyperparameters':{'GBM':{},'CAT':{},'XGB':{}, 'RF':{} ,'LR':{}}},
        "TemporalFusionTransformerModel":{}
       
    })

## the models can be loaded from autgluon directory after the run is completed as it takes almost overnight to run models with best quality
##predictor = TimeSeriesPredictor.load("./AutogluonModels/ag-20230816_215747/")
eval = predictor.evaluate(train_set)
print(eval)
leaderboard_df = predictor.leaderboard(train_set, silent=True)
print(leaderboard_df)
predictions = predictor.predict(train_set)
print(predictions.head())

predictions.to_pickle('./ts_data/bond_predictions_time_series.pkl')

"""Use case2: Equity features only"""

df1['dates'] = pd.to_datetime(df1['dates'].astype(str), format='%Y%m')
equity_features = ['profitability', 'profitability_change',' interest-to-debt:','equity_market_cap','debt_to_equity',
                   'firm_total_debt','book_leverage','ret_6_1_ind','market_leverage','turover_volatility',
                   'debt_to_ebitda','book_to_market', 'earnings_to_price']
equity_features_d_t = ['cusip']+['dates'] + equity_features
df_joint = df1[df1.columns.intersection(equity_features_d_t)]
df_joint['dates'] = pd.to_datetime(df['dates'])
df = df1[df1.columns.intersection(bond_features_d_t)]
# Sort the data by date in ascending order within each panel (CUSIP ID)
drop_cusip = ['98978VAB9', '98978VAH6'] ##columsn have null values
df = df[~df['cusip'].isin(drop_cusip)]
data = df.sort_values(by=['cusip', 'dates'])

# Group the data by CUSIP ID
grouped = data.groupby('cusip')

# Create empty DataFrames to store train and test sets
train_set = pd.DataFrame(columns=data.columns)
test_set = pd.DataFrame(columns=data.columns)

# Iterate through each group (CUSIP ID) and Check if the time series length is at least 4 years so that there is enough
## length to do the validation, window and test. However, it can be done directly with Autogluon but we did it here by ourself because autogluon sometimes
## do not cut test data properly and since time series it has to be cut properly
for cusip, group_data in grouped:
    
    if len(group_data) >= 12 * 4:  # Assuming monthly data
        # Get the last 10 months' data for the test set
        test_data = group_data.tail(12)
        test_set = test_set.append(test_data)
        
        # Get all data for the train set because autogluon you specify prediction length which later becomes 
        ## outof sample data so it can be match with test data
        ##train_data = group_data.drop(test_data.index)
        train_set = train_set.append(group_data)

""" Please note that this is single window. However, we did multiple windows with rolling and fix and this can be done from the below code
 just uncomment if you want to do it also in fit you need to add extra param called num_val_windows. However, wit windows
 run time increase exponentially from 1 day to 3 days"""
#splitter = MultiWindowSplitter(num_windows=5)
#train_data, test_data_multi_window = splitter.split(test_data, prediction_length)
#
#predictor.evaluate(test_data_multi_window)
# Display information about the train and test sets
print("Train Set:")
print(train_set.head())
print("\nTest Set:")
print(test_set.head())
## Similar to tb_model autogluon does it it for you but you have to know the params or uncomment the below 2 lines
#train_set.interpolate(method='bfill', inplace=True)
#test_set.interpolate(method='bfill', inplace=True)
#df['dates'] = pd.to_datetime(df['dates'])
#
#df.interpolate(method='bfill', inplace=True)
train_set['dates'] = pd.to_datetime(train_set['dates'])
test_set['dates'] = pd.to_datetime(test_set['dates'])
prediction_length = 12 ### can be changed it is monthing but the months you want to predict
# Convert 'Time' column to datetime format
#df['dates'] = pd.to_datetime(df['dates'])
## this is time series format for Autogluon to do the modelling where you can say static feature like id in our case
## it is cusip
train_time_series = TimeSeriesDataFrame.from_data_frame(
    train_set,
    id_column="cusip",
    timestamp_column="dates"
)
print(train_time_series.freq)

## the steps we did before can be done with this one single line below but has to be done carefully
## and similarly to tb validation is done automatically by autogluon just speciy the years
#train_data, test_data = train_time_series.train_test_split(prediction_length)
## In the predictor you can specify known covariates i.e. the values you know in future where the past covariates 
## you do not have to specify because they are the variables picked by you
predictor = TimeSeriesPredictor(prediction_length=prediction_length,
    target="excreturn",
    eval_metric=  "RMSE",
    verbosity=4,
    ignore_time_index=True)
predictor.fit(train_time_series,   hyperparameters={
        "DirectTabularModel": {'tabular_hyperparameters':{'GBM':{},'CAT':{},'XGB':{}, 'RF':{} ,'LR':{}}},
        "TemporalFusionTransformerModel":{}
       
    })

## the models can be loaded from autgluon directory after the run is completed as it takes almost overnight to run models with best quality
##predictor = TimeSeriesPredictor.load("./AutogluonModels/ag-20230816_215747/")
eval = predictor.evaluate(train_set)
print(eval)
leaderboard_df = predictor.leaderboard(train_set, silent=True)
print(leaderboard_df)
predictions = predictor.predict(train_set)
print(predictions.head())

predictions.to_pickle('./ts_data/bond_predictions_time_series_equity.pkl')

"""Use Case3: Use both equity, bond and macro features"""

df1['dates'] = pd.to_datetime(df1['dates'].astype(str), format='%Y%m')
all_features = ['cusip']+['dates'] + equity_features + bond_macro_features
df_joint = df1[df1.columns.intersection(all_features)]
df_joint['dates'] = pd.to_datetime(df['dates'])
df = df1[df1.columns.intersection(bond_features_d_t)]
# Sort the data by date in ascending order within each panel (CUSIP ID)
drop_cusip = ['98978VAB9', '98978VAH6'] ##columsn have null values
df = df[~df['cusip'].isin(drop_cusip)]
data = df.sort_values(by=['cusip', 'dates'])

# Group the data by CUSIP ID
grouped = data.groupby('cusip')

# Create empty DataFrames to store train and test sets
train_set = pd.DataFrame(columns=data.columns)
test_set = pd.DataFrame(columns=data.columns)

# Iterate through each group (CUSIP ID) and Check if the time series length is at least 4 years so that there is enough
## length to do the validation, window and test. However, it can be done directly with Autogluon but we did it here by ourself because autogluon sometimes
## do not cut test data properly and since time series it has to be cut properly
for cusip, group_data in grouped:
    
    if len(group_data) >= 12 * 4:  # Assuming monthly data
        # Get the last 10 months' data for the test set
        test_data = group_data.tail(12)
        test_set = test_set.append(test_data)
        
        # Get all data for the train set because autogluon you specify prediction length which later becomes 
        ## outof sample data so it can be match with test data
        ##train_data = group_data.drop(test_data.index)
        train_set = train_set.append(group_data)

""" Please note that this is single window. However, we did multiple windows with rolling and fix and this can be done from the below code
 just uncomment if you want to do it also in fit you need to add extra param called num_val_windows. However, wit windows
 run time increase exponentially from 1 day to 3 days"""
#splitter = MultiWindowSplitter(num_windows=5)
#train_data, test_data_multi_window = splitter.split(test_data, prediction_length)
#
#predictor.evaluate(test_data_multi_window)
# Display information about the train and test sets
print("Train Set:")
print(train_set.head())
print("\nTest Set:")
print(test_set.head())
## Similar to tb_model autogluon does it it for you but you have to know the params or uncomment the below 2 lines
#train_set.interpolate(method='bfill', inplace=True)
#test_set.interpolate(method='bfill', inplace=True)
#df['dates'] = pd.to_datetime(df['dates'])
#
#df.interpolate(method='bfill', inplace=True)
train_set['dates'] = pd.to_datetime(train_set['dates'])
test_set['dates'] = pd.to_datetime(test_set['dates'])
prediction_length = 12 ### can be changed it is monthing but the months you want to predict
# Convert 'Time' column to datetime format
#df['dates'] = pd.to_datetime(df['dates'])
## this is time series format for Autogluon to do the modelling where you can say static feature like id in our case
## it is cusip
train_time_series = TimeSeriesDataFrame.from_data_frame(
    train_set,
    id_column="cusip",
    timestamp_column="dates"
)
print(train_time_series.freq)

## the steps we did before can be done with this one single line below but has to be done carefully
## and similarly to tb validation is done automatically by autogluon just speciy the years
#train_data, test_data = train_time_series.train_test_split(prediction_length)
## In the predictor you can specify known covariates i.e. the values you know in future where the past covariates 
## you do not have to specify because they are the variables picked by you
predictor = TimeSeriesPredictor(prediction_length=prediction_length,
    target="excreturn",
    eval_metric=  "RMSE",
    verbosity=4,
    ignore_time_index=True)
predictor.fit(train_time_series,   hyperparameters={
        "DirectTabularModel": {'tabular_hyperparameters':{'GBM':{},'CAT':{},'XGB':{}, 'RF':{} ,'LR':{}}},
        "TemporalFusionTransformerModel":{}
       
    })

## the models can be loaded from autgluon directory after the run is completed as it takes almost overnight to run models with best quality
##predictor = TimeSeriesPredictor.load("./AutogluonModels/ag-20230816_215747/")
eval = predictor.evaluate(train_set)
print(eval)
leaderboard_df = predictor.leaderboard(train_set, silent=True)
print(leaderboard_df)
predictions = predictor.predict(train_set)
print(predictions.head())

predictions.to_pickle('./ts_data/bond_predictions_time_series_equity.pkl')


