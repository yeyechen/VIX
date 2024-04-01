from iVIX import MEANINGFUL_VIX_INDEX_MATURITY_THRESHOLD
from VIX_2020_minute import CONSTANT_MATURITY_TERM
import VIX_2020_minute
import os
import pandas as pd


def main():
    # calculate vix index and implied volatility for every available trading time
    ivix = []
    trading_time = sorted(new_options_dataset.index.unique())

    # calculating vix
    for (i, time) in enumerate(trading_time):
        ivix.append(VIX_2020_minute.calcVixIndex(time, new_options_dataset))
        # progressBar(len(trading_time), i)

    # output to a csv file
    output = pd.DataFrame({'trading_time': trading_time, 'vix': ivix})
    save_folder = 'calc_data_daily'
    file_name = 'daily_cmt=' + str(CONSTANT_MATURITY_TERM) + '_thresh=' + str(MEANINGFUL_VIX_INDEX_MATURITY_THRESHOLD) + '.csv'
    save_path = os.path.join(file_path, save_folder, file_name)
    output.to_csv(save_path, index=False)


if __name__ == '__main__':

    file_path = os.getcwd()

    new_options_file_name = 'option_total_year_2020_daily.csv'
    new_options_file_path = os.path.join(file_path, new_options_file_name)
    new_options_dataset = pd.read_csv(new_options_file_name, encoding='GBK')

    # adjust the structure of options dataset to fit in original code
    # change: setting 'vix_time' as index, selecting useful columns, renaming 'expire_date' to 'EXE_ENDDATE'
    new_options_dataset['time'] = new_options_dataset['time'].astype(str).str.split(' ').str[-1]
    new_options_dataset['vix_time'] = pd.to_datetime(new_options_dataset['date'] + ' ' + new_options_dataset['time'])
    new_options_dataset = new_options_dataset.set_index('vix_time')
    selected_columns = ['price_call', 'price_put', 'k_x', 'd2e', 'timeleft', 'r', 'expire_date', 'iv']
    new_options_dataset = new_options_dataset[selected_columns]
    new_options_dataset = new_options_dataset.rename(columns={'expire_date': 'EXE_ENDDATE'})
    trading_close_time = '15:00'
    new_options_dataset['EXE_ENDDATE'] = new_options_dataset['EXE_ENDDATE'].str.replace('-', '/') + ' ' + trading_close_time

    main()

