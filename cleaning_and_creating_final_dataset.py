import pandas as pd
#from autoviz.AutoViz_Class import AutoViz_Class
#%matplotlib_inline
import matplotlib as plt
import numpy as np
import scipy

"""This function clean the FISD data provided by the Prof. Steffen and clean it 
   and take all relevant columns and saves it in data directory"""

def clean_fisd(filename):
    fisd_main= pd.read_csv(filename)
    #print('Issues in FISD', len(fisd_main)) 
    #print (len(fisd_main.prospectus_issuer_name.unique()))
    fisd_main = fisd_main[(fisd_main['bond_type'] =='CDEB') | (fisd_main['bond_type'] =='CPIK') |
                          (fisd_main['bond_type'] =='CZ') | (fisd_main['bond_type'] =='USBN') |
                          (fisd_main['bond_type'] =='CMTN') | (fisd_main['bond_type'] =='CMTZ')]
    fisd_main = fisd_main[fisd_main['foreign_currency'] =='N']
    fisd_main = fisd_main[(fisd_main['pay_in_kind'] !='Y') | (fisd_main['pay_in_kind'].isna()) ]
    fisd_main = fisd_main[fisd_main['pay_in_kind_exp_date'].isna()]
    fisd_main = fisd_main[(fisd_main['yankee'] =='N') | (fisd_main['yankee'].isna()) ]
    fisd_main = fisd_main[(fisd_main['canadian'] =='N') | (fisd_main['canadian'].isna()) ]
    fisd_main = fisd_main[(fisd_main['coupon_type'] =='F') | (fisd_main['coupon_type']=='Z') ]
    fisd_main = fisd_main[fisd_main['fix_frequency'].isna()]
    fisd_main = fisd_main[fisd_main['coupon_change_indicator']=='N']
    fisd_main = fisd_main[(fisd_main['interest_frequency'] ==0) | (fisd_main['interest_frequency'] ==1) |
                           (fisd_main['interest_frequency'] ==2) | (fisd_main['interest_frequency'] ==4) | (fisd_main['interest_frequency'] ==12)]
    fisd_main = fisd_main[fisd_main['rule_144a']=='N']
    fisd_main = fisd_main[(fisd_main['private_placement'] =='N') | (fisd_main['private_placement'].isna()) ]
    fisd_main = fisd_main[fisd_main['defaulted']=='N']
    fisd_main = fisd_main[fisd_main['filing_date'].isna()]
    fisd_main = fisd_main[fisd_main['convertible']=='N']
    fisd_main = fisd_main[fisd_main['exchange'].isna()]
    fisd_main = fisd_main[(fisd_main['putable'] =='N') | (fisd_main['putable'].isna()) ]
    fisd_main = fisd_main[(fisd_main['unit_deal'] =='N') | (fisd_main['unit_deal'].isna()) ]
    fisd_main = fisd_main[(fisd_main['exchangeable'] =='N') | (fisd_main['exchangeable'].isna()) ]
    fisd_main = fisd_main[fisd_main['perpetual'] =='N' ]
    fisd_main = fisd_main[(fisd_main['preferred_security'] =='N') | (fisd_main['preferred_security'].isna()) ]
    fisd_main['cusip_id'] = fisd_main['issuer_cusip'] + fisd_main['issue_cusip']
    return fisd_main

### transform merged issue ###
# filename = 'mergedissue.csv'
# fisd_main = clean_fisd(filename)
# fisd_main.to_parquet('fisd.parquet', engine="fastparquet",  compression='brotli' )

### read tranformed merged isue ###
fisd_main = pd.read_parquet('./data/fisd.parquet')
good_cusips = pd.DataFrame(fisd_main['cusip_id'])
good_cusips['identifier'] = 1
print (len(good_cusips))
print ('Issues in FISD', len(fisd_main))


""" Trace Enhanced and Equity Financial ratios Data Downloaded from WRDS 

    https://wrds-www.wharton.upenn.edu/pages/get-data/otc-corporate-bond-and-agency-debt-bond-transaction-data/trace-enhanced/bond-trades/
    This function clean the trace data and perform the Dick-Nielsen cleaning while merging it with FISD, and calculating bond features
    Please note do not download all the fields from trace otherwise it will not load in memory use only
    cusip, trade date, price, volume, as_of_cd, orig_msg_seq_nb and trc_st"""
    
def clean_trace():
    trace = pd.read_csv('./trace.csv')
    ## check trace data types
    #print(trace.dtypes)
    #print( trace.memory_usage(deep=True))
    ### common cusipid between transformed merged issue and trace trade are ###
    common_cusip_id = np.intersect1d(fisd_main['cusip_id'], trace['cusip_id'].astype(str))
    #print('length of common cusip_id are', len(common_cusip_id))

    # #changing date to datetime
    trace['trd_exctn_dt'] = trace.trd_exctn_dt.str[:4] + '-' + trace.trd_exctn_dt.str[5:7] + '-' + trace.trd_exctn_dt.str[8:10]
    trace['trd_exctn_dt'] = pd.to_datetime(trace['trd_exctn_dt'])
    print ('Dataset runs from', trace['trd_exctn_dt'].min(), 'to', trace['trd_exctn_dt'].max())

    # #changing trade size format so we can operate on sizes as floats
    trace.ascii_rptd_vol_tx[trace.ascii_rptd_vol_tx =='1MM+' ] =  1000005
    trace.ascii_rptd_vol_tx[trace.ascii_rptd_vol_tx =='5MM+' ] =  5000005
    trace['ascii_rptd_vol_tx'] = trace['ascii_rptd_vol_tx'].astype(float)
    #print(len(trace), trace.asof_cd.unique(), trace.trc_st.unique())
    #print(trace.dtypes)
    #print( trace.memory_usage(deep=True))
    #print ('unique trace cusips',len(trace['cusip_id'].unique()))
    #print ('total trades',len(trace))
    #print ('unique fisd len',len(good_cusips),'(same as above)')
    trace = pd.merge(trace,good_cusips,on=['cusip_id'],how = 'inner')
    #print(trace.head())
    trace = trace[pd.notnull(trace['rptd_pr'])]
    #print ('Unique CUSIPs in overlapping TRACE/FISD datasets',len(trace['cusip_id'].unique()))
    #print ('Unique Trades',len(trace))

    """Applying Dick-Nielsen - Part 1
    The first part of the filter deletes trades that were canceled on the same day as 
    the original transaciton. Same day cancels are identified by the TRC_ST column.
    The logic is:
    If TRC_ST = H or C then delete it
    Also delete the trade from that day whose MSG_SEQ_NB equals the deleted H or C trade's ORIG_MSG_SEQ_NB
    IF TRC_ST = I or W, then delete the I or W trade.
    I or W means that the original trade was updated to reflect the error so deleting this fixes the problem."""
    trace_len_pre_filter = len(trace)
    #print ('pre filter length', trace_len_pre_filter)

    #delete I/W
    trace = trace[(trace['trc_st'] != 'I')&(trace['trc_st'] != 'W')]
    #print ('post I/W delete length', trace_len_pre_filter)

    #create dataframe of H/C trades
    trace_same_day_cancel = trace[(trace['trc_st'] == 'H') | (trace['trc_st'] == 'C')]
    #print(trace_same_day_cancel)
    trace_same_day_cancel = trace_same_day_cancel[['cusip_id','trd_exctn_dt','orig_msg_seq_nb']]
    #print(trace_same_day_cancel)
    trace_same_day_cancel['cancel_trd'] = 1
    #trace_same_day_cancel = trace_same_day_cancel.rename(columns={'orig_msg_seq_nb':'MSG_SEQ_NB'})

    trace = pd.merge(trace,trace_same_day_cancel,on=['cusip_id','trd_exctn_dt','orig_msg_seq_nb'],how = 'outer')
    trace = trace[(trace['trc_st'] != 'H') & (trace['trc_st'] != 'C')]
    trace = trace[pd.notnull(trace['trc_st'])]
    trace = trace[(trace['cancel_trd'] != 1)]
    trace = trace[(pd.notnull(trace['trd_exctn_dt']))]
    trace.drop(['trc_st', 'cancel_trd', 'orig_msg_seq_nb'], axis=1,inplace = True)
    #len trace should decline by> trace_same_day_cancel
    #print ('final length', len(trace))

    """The second part of the filter deletes trades that were canceled on day different from 
    the original transaction. Different day cancels are identified by the ASOF_CD column.
    The logic is:
    If ASOF_CD = X then delete
    Canceled trade
    If ASOF_CD = R, then
    Delete R trade
    Delete prior day trade with same CUSIP/price/size"""

    trace = trace[(trace['asof_cd'] != 'X')]

    trace_diff_day_cancel = trace[(trace['asof_cd'] == 'R')]
    trace = trace[(trace['asof_cd'] != 'R')]

    trace_diff_day_cancel = trace_diff_day_cancel[['cusip_id','rptd_pr','ascii_rptd_vol_tx']]
    trace_diff_day_cancel['cancel_trd'] = 1
    ### Merge the FISD with Trace to get common bond characterstics like coupon , and face value 
    trace = pd.merge(trace,trace_diff_day_cancel,on=['cusip_id','rptd_pr','ascii_rptd_vol_tx'],how = 'outer')
    trace = trace[(trace['asof_cd'] != 'R')]
    trace = trace[(trace['cancel_trd'] != 1)]
    trace = trace[(pd.notnull(trace['trd_exctn_tm']))]
    trace.drop(['asof_cd', 'cancel_trd'], axis=1,inplace = True)
    #print (len(trace))

    fisd_for_yield = fisd_main[['cusip_id','maturity','coupon','interest_frequency','principal_amt', 'treasury_spread']]
    fisd_for_yield['maturity'] = pd.to_datetime(fisd_for_yield['maturity'])
    trace = trace.merge(fisd_for_yield,on='cusip_id',how='inner')
    trace = trace[pd.notnull(trace['rptd_pr'])]
    trace['days_to_maturity'] = (trace['maturity'] - trace['trd_exctn_dt']).astype('timedelta64[D]')
    trace['n_maturity'] = (trace['days_to_maturity'] / 365) * 2
    trace = trace[trace['days_to_maturity'] > 0]
    #print (len(trace))
    #print(trace.head())
    #print(trace["principal_amt"].unique())
    """ Sanity check for yield"""   
    def Px(Rate,Mkt_Price,Face,Freq,N,C):
        return Mkt_Price - (Face * ( 1 + Rate / Freq ) ** ( - N ) + ( C / Rate ) * ( 1 - (1 + ( Rate / Freq )) ** -N ) )

    def YieldCalc(guess,Mkt_Price,Face,Freq,N,C):
        x = scipy.optimize.newton(Px, guess,args = (Mkt_Price,Face,Freq,N,C), tol=.0000001, maxiter=100)
        return x
    yld = {}
    for i in trace.index:
        try:
            yld.update({i:YieldCalc(trace['coupon'].loc[i]/(trace['principal_amt'].loc[i]/10),trace['rptd_pr'].loc[i],
                                    trace['principal_amt'].loc[i]/10, trace['interest_frequency'].loc[i],
                                    trace['n_maturity'].loc[i], trace['coupon'].loc[i])})
        except(RuntimeError):
            pass
        else:
            pass
    trace['ytm']=pd.Series(yld)
    trace = trace.sort_values(by = ['cusip_id','trd_exctn_dt','trd_exctn_tm'])
    #print(trace.head())
    trace = trace[trace['ascii_rptd_vol_tx'] > 99999]
    #print(len(trace))
    trace.set_index("trd_exctn_dt", inplace =True)
    trace.sort_index(inplace=True)
    # Create an empty DataFrame to store the results
    trace_final = pd.DataFrame()
    # Iterate over unique bond identifiers (e.g., CUSIP) to get monthly prices from daily prices and calculate returns
    unique_bond_ids = trace['cusip_id'].unique()
    for bond_id in unique_bond_ids:
        # Filter data for the current bond identifier
        bond_data_subset = trace[trace['cusip_id'] == bond_id]
        final_data = pd.DataFrame()
        # Set the month-end trading price to the last available daily price of the month
        final_data['month_end_price'] = bond_data_subset['rptd_pr'].resample('M').apply(lambda x: x[-1])
        if bond_data_subset["interest_frequency"].unique()[0] ==2:
            final_data['coupon'] = 0
            final_data['coupon'].iloc[6::6] = bond_data_subset['coupon'].unique()[0]
            final_data['accured_interest'] = final_data['coupon'].cumsum()
        if bond_data_subset["interest_frequency"].unique()[0] ==4:
            final_data['coupon'] = 0
            final_data['coupon'].iloc[4::4] = bond_data_subset['coupon'].unique()[0]
            final_data['accured_interest'] = final_data['coupon'].cumsum()

        # Calculate monthly returns
        num_return = final_data['month_end_price'] + final_data['coupon'] + final_data['accured_interest']
        denom_return = final_data['month_end_price'].shift(1) + final_data['accured_interest'].shift(1)
        final_data['monthly_returns'] =  (num_return /denom_return)- 1
        ## cusip ##
        final_data['cusip_id'] =bond_data_subset['cusip_id'].unique()[0]
        ## Bond Age
        first_date = final_data.index[0]
        date_index_years = [(date.year - first_date.year) + (date.month - first_date.month) / 12 for date in final_data.index]
        final_data['age']=date_index_years
        final_data['age']=final_data['age'].round(2)
        ## face_value ##
        final_data['face_value']=bond_data_subset.principal_amt.unique()[0]
        ## momentum 6 month/ short term reversal##
        final_data['6_month_momentum'] = final_data['monthly_returns'].rolling(window=6).sum()
        ## rolling volatility, skewness and kurtosis over 12 month
        final_data['volatility'] = final_data['monthly_returns'].rolling(window=12).std()
        final_data['volatility'] = final_data['monthly_returns'].rolling(window=12).skew()
        final_data['volatility'] = final_data['monthly_returns'].rolling(window=12).kurt()
        ### Lagged Returns ##
        final_data['one_month_lag'] = final_data['monthly_returns'].shift(1)
        final_data['two_month_lag'] = final_data['monthly_returns'].shift(2)
        final_data['three_month_lag'] = final_data['three_returns'].shift(3)
        final_data["time_to_maturity"] = date_index_years[::-1]
        final_data["spread"] = bond_data_subset.treasury_spread.unique()[0]
        final_data = final_data.drop(['month_end_price', 'coupon', 'accured_interest'], axis=1)
        final_data['coupon'] = bond_data_subset.coupon.unique()[0]
        trace_final = pd.concat([trace_final, final_data])

    return final_data
bond_features = clean_trace()
bond_features = bond_features.reset_index().rename(columns={'trd_exctn_dt': 'dates'})

"""read the equity ratios data downloaded from WRDS finanacial ratios and can be downloaded from
   https://wrds-www.wharton.upenn.edu/pages/get-data/financial-ratios-suite-wrds/
   please note that it is possible that you have to do some manipluation before merging it with trace_datait depends on the download
   also change the name from cusip to cusip_id in financial_ratio data and the other names required, in this case it is done manually
   therefore, merged directly"""
f_ratios = pd.read_csv("f_ratio_wrds.csv")

features_data= bond_features.merge(f_ratios, on = 'cusip_id', how= 'left')
""" read the macro data which can downloaded from 
    3months_treasury = https://fred.stlouisfed.org/series/DTB3
    term_spread = https://fred.stlouisfed.org/series/T10Y3M
    inflation = https://fred.stlouisfed.org/series/FPCPITOTLZGUSA
    the Tbill and term spread are daily data but in the website it can be changed to monthly with the computation method
    we choose the average , one can do it by themselve by resampling like we did above for other features
    However, for inflnation it is annual and therefore has to be matched with dates, lastly all the series generates some headers 
    then can be cleaned manually or automatically we cleaned it manually to understand the data as well as its usage for us
    """
t_bill = pd.read_csv("t_bill.csv")
t_bill = t_bill.rename(columns={'observation_date': 'dates', 'DTB3': 't_bill'})
term_spread = pd.read_csv("term_spread.csv")
term_spread = term_spread.rename(columns={'observation_date': 'dates', 'T10Y3M': 'term_spread'})
inflation =pd.read_csv("inflation.csv")
inflation = inflation.rename(columns={'observation_date': 'dates', 'FPCPITOTLZGUSA': 'inflation'})
features_data = features_data.merge(t_bill, on = 'dates', how= 'left')
features_data = features_data.merge(term_spread, on = 'dates', how= 'left')
features_data['Year'] = features_data['dates'].dt.year
inflation['Year'] = inflation['dates'].dt.year
features_data = features_data.merge(inflation, on='Year', how='left')
# Drop the redundant 'Year' column
features_data.drop('Year', axis=1, inplace=True)
features_data['excreturn'] = features_data['monthly_returns'] - features_data['t_bill']

""" Finally Read the ratings data downloaded from WRDS finanacial ratios and can be downloaded from
https://wrds-www.wharton.upenn.edu/pages/get-data/compustat-capital-iq-standard-poors/compustat/north-america-daily/ratings/"""
ratings_df = pd.read_csv("ratings.csv")
ratings_df = ratings_df.drop(['gvkey', 'datadate'], axis=1)
ratings_df = ratings_df.dropna()
# Define a mapping of S&P ratings to numbers
rating_mapping = {
    'AAA': 1,
    'AA+': 2,
    'AA': 3,
    'AA-': 4,
    'A+': 5,
    'A': 6,
    'A-': 7,
    'BBB+': 8,
    'BBB': 9,
    'BBB-': 10,
    'BB+': 11,
    'BB': 12,
    'BB-': 13,
    'B+': 14,
    'B': 15,
    'B-': 16,
    'CCC+': 17,
    'CCC': 18,
    'CCC-': 19,
    'CC': 20,
    'C': 21,
    'D': 22
}
# Map S&P ratings to numbers in the DataFrame
ratings_df = ratings_df.rename(columns={'cusip': 'cusip_id', 'splticrm': 'ratings'})
ratings_df['rating'] = ratings_df['rating'].map(rating_mapping)
features_data= features_data.merge(f_ratios, on = 'cusip_id', how= 'left')

features_data.to_pickle('./data/final_features.pkl')










     






