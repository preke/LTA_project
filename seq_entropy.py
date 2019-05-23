import numpy as np
import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score

data_path_list = [
    'DR0008_activity_accumulator_2016_09.csv',
    'DR0008_activity_accumulator_2016-10.csv',
    'DR0008_activity_accumulator_2016-11.csv',
    'DR0008_activity_accumulator_2016-12.csv'
]

def create_day_seq(days, length):
    tmp_dict = {}
    for day in days:
        try:
            tmp_dict[day] += 1
        except:
            tmp_dict[day] = 1
    res = [0]*length
    for k,v in tmp_dict.items():
        res[k] = v
    return res

def extract_function_seq(data_path, function, month='9', within_day=False):
    df                  = pd.read_csv(data_path, sep='\t')
    df_temp             = df[df['event_type'] == function][['De-id', 'timestamp']]
    df_temp['day']      = df_temp['timestamp'].apply(lambda x: x.day)
    df_day_list         = df_temp[['De-id', 'day']].groupby('De-id').agg(create_day_seq, length=df_temp['day'].nunique()).reset_index()
    df_day_list.columns = ['De-id', month + '_day_list']
    return df_day_list


def ApEn(U, m, r):

    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def _phi(m):
        x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = [len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (N - m + 1.0) for x_i in x]
        return (N - m + 1.0)**(-1) * sum(np.log(C))

    N = len(U)

    return abs(_phi(m+1) - _phi(m))

def get_all_seq(data_path_list, function):
    first_flag = 1
    for data_path in data_path_list:
        df_day_list = extract_function_seq(data_path, function, data_path.split('.')[0][-2:])
        if first_flag:
            df_all = df_day_list.copy()
            first_flag = 0
        else:
            df_all = pd.merge(df_all, df_day_list, on='De-id', how='left')
    return df_all

if __name__ == '__main__':
    df_all = get_all_seq(data_path_list, function)

    X_train, X_test, y_train, y_test = train_test_split(df_all_entropy, df_all_label, test_size = 0.2)
    brf = BalancedRandomForestClassifier(n_estimators=200, criterion = 'gini', max_features = 1.0, random_state=0)
    brf.fit(X_train, y_train) 
    y_pred = brf.predict(X_test)
    imp_feature = brf.feature_importances_
    print(confusion_matrix(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))
    print(balanced_accuracy_score(y_test, y_pred))


    # df_all['11_day_list'] = df_all['11_day_list'].fillna('0')
    # df_all['11_day_list'] = df_all['11_day_list'].apply(lambda x: x if x !='0' else [0]*32)
    # df_all['09_entropy'] = df_all['09_day_list'].apply(lambda x: ApEn(x, m=2, r=np.mean(x)*0.2))




