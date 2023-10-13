from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd
import os


df1 = pd.read_pickle('./data/final_dataset.pkl')
df1['dates'] = pd.to_datetime(df1['dates'].astype(str), format='%Y%m')
  
""" First Use case: Using only bond and Macro features"""
bond_macro_features = ['age','coupon','ratings','yields','duration','excreturn','one_month_lag','two_month_lag'
'three_month_lag','spread','6_month_momentum','volatility','skew','VaR' , 'mom6mspread', 'mom6xrtg', 'mom6industry', 'face_value',
'time_to_maturity', 't_bill', 'term_spread', 'inflation']
bond_features_d_t = ['cusip']+['dates'] + bond_macro_features
df = df1[df1.columns.intersection(bond_features_d_t)]

## Autogluon fills the missing values before doing the analysis but it is possible if there nan values it can give error because certain params 
## has to be set but an easy hack to run script to uncomment below line
#df.interpolate(method='bfill', inplace=True)
df['dates'] = pd.to_datetime(df['dates'])

### As you can validation is commented out because autogluon takes it automatically or you can set it by yourself 
## but internally it is better because later when expanding and rooling window will be used no extra loops have to be set up also the windows are 
## set up Autogluon automatically as it is tabular data
train_start = pd.to_datetime('2002-04-01')
train_end = pd.to_datetime('2015-12-01')
##val_start = pd.to_datetime('2013-01-01')
# #val_end = pd.to_datetime('2018-12-01')
test_start = pd.to_datetime('2016-01-01')
test_end = pd.to_datetime('2020-06-01')

train_data = df[(df['dates'] >= train_start) & (df['dates'] <= train_end)]
##val_data = df[(df['dates'] >= val_start) & (df['dates'] <= val_end)]
test_data = df[(df['dates'] >= test_start) & (df['dates'] <= test_end)]
# Print the train, validation, and test data
predict_column = 'excreturn'
print("Summary of return variable: \n", train_data[predict_column].describe())
### Please note that best_quality means all models from linear regression, Random forest to weighted assembling. However, the running time
## will increase exponentially because there is window and big hyperparameter grid search
predictor = TabularPredictor(label=predict_column).fit(train_data, presets='best_quality')
## Models are stored by default in autogluonmodels directory and then you can load all or best model or each model one by one by the below commented line
#predictor = TabularPredictor.load("./AutogluonModels/ag-20230814_204701/")
performance = predictor.predict(test_data)
results = predictor.fit_summary(show_plot=True)
print("AutoGluon identified the following types of features:")
print(predictor.feature_metadata)
print("Best model is: ", predictor.get_model_best())
print(performance = predictor.evaluate(test_data))
print(predictor.leaderboard(test_data, silent=True))
feature_imp_bond = predictor.feature_importance(test_data)
print(feature_imp_bond)
feature_imp_bond.to_csv('feature_imp_bond_only.csv')

""" Second Use case: using only equity feature"""
equity_features = ['profitability', 'profitability_change',' interest-to-debt:','equity_market_cap','debt_to_equity',
                   'firm_total_debt','book_leverage','ret_6_1_ind','market_leverage','turover_volatility',
                   'debt_to_ebitda','book_to_market', 'earnings_to_price']
equity_features_d_t = ['cusip']+['dates'] + equity_features
df_joint = df1[df1.columns.intersection(equity_features_d_t)]
df_joint.interpolate(method='bfill', inplace=True)
df_joint['dates'] = pd.to_datetime(df['dates'])

train_data_joint = df_joint[(df_joint['dates'] >= train_start) & (df_joint['dates'] <= train_end)]
##val_data = df[(df['dates'] >= val_start) & (df['dates'] <= val_end)]
test_data_joint = df_joint[(df_joint['dates'] >= test_start) & (df_joint['dates'] <= test_end)]
predictor_joint = TabularPredictor(label=predict_column).fit(train_data_joint, presets='best_quality')
#predictor_joint = TabularPredictor.load("./AutogluonModels/ag-20230815_020008/")
performance_joint = predictor_joint.predict(test_data_joint)
performance.to_frame().to_csv('predictions.csv')
performance_joint.to_frame().to_csv('predictions_joint.csv')
print(results = predictor.fit_summary(show_plot=True))
print("Best model is: ", predictor_joint.get_model_best())
performance = predictor_joint.evaluate(test_data_joint)
print(performance)
print(predictor_joint.leaderboard(test_data_joint, silent=True))
feature_imp_bond_equity = predictor_joint.feature_importance(test_data_joint)
print(feature_imp_bond_equity)
feature_imp_bond_equity.to_csv('feature_imp_bond_equity.csv')

""" Third Use case: Using only Bond, Macro and Equity features"""
# equity_features = ['vixbeta', 'me','be_me','ret_6_1','ni_me','rvol_21d','at_be',
# 'ret_6_1_ind','totaldebt','gp_at','debt_ebitda','at_me','oper_lvg','chg_gp_at', 'turn_vol']
equity_features = ['profitability', 'profitability_change',' interest-to-debt:','equity_market_cap','debt_to_equity',
                   'firm_total_debt','book_leverage','ret_6_1_ind','market_leverage','turover_volatility',
                   'debt_to_ebitda','book_to_market', 'earnings_to_price']
joint_features = bond_features_d_t + equity_features
df_joint = df1[df1.columns.intersection(joint_features)]
df_joint.interpolate(method='bfill', inplace=True)
df_joint['dates'] = pd.to_datetime(df['dates'])

train_data_joint = df_joint[(df_joint['dates'] >= train_start) & (df_joint['dates'] <= train_end)]
##val_data = df[(df['dates'] >= val_start) & (df['dates'] <= val_end)]
test_data_joint = df_joint[(df_joint['dates'] >= test_start) & (df_joint['dates'] <= test_end)]
predictor_joint = TabularPredictor(label=predict_column).fit(train_data_joint, presets='best_quality')
#predictor_joint = TabularPredictor.load("./AutogluonModels/ag-20230815_020008/")
performance_joint = predictor_joint.predict(test_data_joint)
performance.to_frame().to_csv('predictions.csv')
performance_joint.to_frame().to_csv('predictions_joint.csv')
print(results = predictor.fit_summary(show_plot=True))
print("Best model is: ", predictor_joint.get_model_best())
performance = predictor_joint.evaluate(test_data_joint)
print(performance)
print(predictor_joint.leaderboard(test_data_joint, silent=True))
feature_imp_bond_equity = predictor_joint.feature_importance(test_data_joint)
print(feature_imp_bond_equity)
feature_imp_bond_equity.to_csv('feature_imp_bond_equity.csv')








