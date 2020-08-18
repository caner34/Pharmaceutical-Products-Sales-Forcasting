import pandas as pd
import numpy as np
from pytrends.request import TrendReq
import time
import os
from pytrends.exceptions import ResponseError
import random

#RESOURCE: 
#          Google Trends Python Library
#          https://pypi.org/project/pytrends/



def related_queries(list_of_keywords):
    # Related Queries, returns a dictionary of dataframes
    pytrend = TrendReq()
    pytrend.build_payload(kw_list=list_of_keywords)
    related_queries = pytrend.related_queries()
    return related_queries #.values()


# related_queries_result = related_queries(['korona'])



            
def get_istanbul_coefficent_series_as_reference(keyword):
    pytrend = TrendReq()
    # 'TR-34' as geo parameter returns the data for the city Istanbul
    pytrend.build_payload(
         kw_list=[keyword],
         cat=0,
         timeframe='2017-01-01 2020-05-01',
         geo='TR-34',
         gprop='')
    data_ist_45_m = pytrend.interest_over_time()
    
    data_ist_45_m = data_ist_45_m.reset_index()
    
    data_ist_45_m['date'] = data_ist_45_m['date'].astype(str)
    data_ist_45_m['month'] = data_ist_45_m['date'].apply(lambda x: x.split('-')[1])
    data_ist_45_m['year'] = data_ist_45_m['date'].apply(lambda x: x.split('-')[0])
    data_ist_45_m['Period'] = data_ist_45_m['date'].apply(lambda x: x.split('-')[0]+x.split('-')[1])
    
    coefficients_ist_over_time = data_ist_45_m.groupby(['Period'])[keyword].mean()
    return coefficients_ist_over_time
  

         

def get_timeframe_and_period_string(year, imonth):
    # construct timeframe string
    cr_month_str = str(imonth+1)
    next_month_str = str((imonth+1)%12+1)
    if len(cr_month_str) == 1:
        cr_month_str = '0'+cr_month_str
    if len(next_month_str) == 1:
        next_month_str = '0'+next_month_str
    cr_year_str = str(year)
    next_year_str = str(year)
    if next_month_str == '01':
        next_year_str = str(year+1)
    timeframe_str = cr_year_str+'-'+cr_month_str+'-01 '+next_year_str+'-'+next_month_str+'-01'
    period_str = cr_year_str+cr_month_str
    return timeframe_str, period_str
    


def get_cr_data_portion_of_interest(y, im, keyword, coefficients_ist_over_time):
    pytrend = TrendReq()
    cr_timeframe_str, period_str = get_timeframe_and_period_string(y, im)
    pytrend.build_payload(
         kw_list=[keyword],
         cat=0,
         timeframe=cr_timeframe_str,        #'today 45-m',
         geo='TR',  #'TR-34',
         gprop='')
    cr_data = pytrend.interest_by_region()
    cr_data = cr_data.reset_index()
    cr_data['Period'] = period_str
    # normalize and sync the data within this portion of time with the whole time vector
    cr_data[keyword] = cr_data[keyword].apply(lambda x: x/coefficients_ist_over_time[period_str])
    return cr_data



def gather_interest_on_a_keyword_over_time_by_city(keyword, dest_dir):
    pytrend = TrendReq()
    
    #we will link each piece of monthly time frame based on this reference
    coefficients_ist_over_time = get_istanbul_coefficent_series_as_reference(keyword)
    
    data_result = pd.DataFrame(columns = ['geoName', keyword, 'Period'])
    
    count = 0
    
    years = [2017, 2018, 2019, 2020]
    # retrieve data for each month and year
    for y in years:
        for im in range(0,12):
            if y == 2020 and im > 3:
                break
            
            time.sleep(0.5)
            
            for n in range(0, 5):
                try:
                    cr_data = get_cr_data_portion_of_interest(y, im, keyword, coefficients_ist_over_time)
                    # append the data from current time period with the main data frame
                    data_result = data_result.append(cr_data, ignore_index=True)
                    break
                except ResponseError:
                    # if error.resp.reason in ['userRateLimitExceeded', 'quotaExceeded', 'internalServerError', 'backendError']:
                    print(ResponseError.mro())
                    time.sleep((2 ** n) + random.random())
                    # else:
                    #     break

            count += 1
            print(count)
    
    # save dataframe to the destination
    full_file_path = dest_dir+'trends_keyword_'+keyword.replace(' ', '_').replace('ç', 'c').replace('ğ', 'g').replace('ü', 'u').replace('ö', 'o').replace('ş', 's').replace('ı', 'i').replace('ç', 'c')+'.csv'
    with open(full_file_path, 'w') as output_file:
        data_result.to_csv(output_file, encoding='utf-8', index=False)
    #data_result.to_csv(os.path.join(dest_dir, r'trends_keyword_'+keyword+'.csv'), encoding='utf-8', index=False)
    print('Succesfully saved into {}'.format(full_file_path))
    return data_result
    


data = gather_interest_on_a_keyword_over_time_by_city('hastane randevu', 
                                               '/home/a/Desktop/AnacondaProjects/Competition2020_ABC/datasets/google_trends/')






