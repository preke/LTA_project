import numpy as np
import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
import re

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
    res = [0]*(length+1)
    for k,v in tmp_dict.items():
        res[k] = v
    return res

def extract_function_seq(data_path, function, month='9', within_day=False):
    df                   = pd.read_csv(data_path, sep='\t')
    df_temp              = df[df['event_type'] == function][['De-id', 'timestamp']]
    df_temp['timestamp'] = df_temp['timestamp'].apply(pd.to_datetime)
    df_temp['day']       = df_temp['timestamp'].apply(lambda x: x.day)
    df_day_list          = df_temp[['De-id', 'day']].groupby('De-id').agg(create_day_seq, length=df_temp['day'].nunique()).reset_index()
    df_day_list.columns  = ['De-id', month + '_day_list']
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

    df_all['09_day_list'] = df_all['09_day_list'].fillna(0)
    df_all['10_day_list'] = df_all['10_day_list'].fillna(0)
    df_all['11_day_list'] = df_all['11_day_list'].fillna(0)
    df_all['12_day_list'] = df_all['12_day_list'].fillna(0)

    df_all['09_day_list'] = df_all['09_day_list'].apply(lambda x: [0]*31 if x == 0 else x)
    df_all['10_day_list'] = df_all['10_day_list'].apply(lambda x: [0]*32 if x == 0 else x)
    df_all['11_day_list'] = df_all['11_day_list'].apply(lambda x: [0]*31 if x == 0 else x)
    df_all['12_day_list'] = df_all['12_day_list'].apply(lambda x: [0]*32 if x == 0 else x)

    df_all['total_list']  = df_all.apply(lambda row: row['09_day_list'][1:] +  row['10_day_list'][1:]
                                           + row['11_day_list'][1:] +  row['12_day_list'][1:], axis=1)
    return df_all

def get_seq_entropy(df_all, m):

    df_all['09_entropy' + '_' + str(m)]    = df_all['09_day_list'].apply(lambda x: ApEn(x, m=m, r=np.mean(x)*0.2))
    df_all['10_entropy' + '_' + str(m)]    = df_all['10_day_list'].apply(lambda x: ApEn(x, m=m, r=np.mean(x)*0.2))
    df_all['11_entropy' + '_' + str(m)]    = df_all['11_day_list'].apply(lambda x: ApEn(x, m=m, r=np.mean(x)*0.2))
    df_all['12_entropy' + '_' + str(m)]    = df_all['12_day_list'].apply(lambda x: ApEn(x, m=m, r=np.mean(x)*0.2))
    df_all['total_entropy' + '_' + str(m)] = df_all['total_list'].apply(lambda x: ApEn(x, m=m, r=np.mean(x)*0.2))
    return df_all

def get_all_seq(data_path_list, function):
    first_flag = 1
    for data_path in data_path_list:
        df_day_list = extract_function_seq(data_path, function, data_path.split('.')[0][-2:])
        if first_flag:
            df_all = df_day_list.copy()
            first_flag = 0
        else:
            df_all = pd.merge(df_all, df_day_list, on='De-id', how='left')

    df_all['09_day_list'] = df_all['09_day_list'].fillna(0)
    df_all['10_day_list'] = df_all['10_day_list'].fillna(0)
    df_all['11_day_list'] = df_all['11_day_list'].fillna(0)
    df_all['12_day_list'] = df_all['12_day_list'].fillna(0)

    df_all['09_day_list'] = df_all['09_day_list'].apply(lambda x: [0]*31 if x == 0 else x)
    df_all['10_day_list'] = df_all['10_day_list'].apply(lambda x: [0]*32 if x == 0 else x)
    df_all['11_day_list'] = df_all['11_day_list'].apply(lambda x: [0]*31 if x == 0 else x)
    df_all['12_day_list'] = df_all['12_day_list'].apply(lambda x: [0]*32 if x == 0 else x)

    df_all['total_list']  = df_all.apply(lambda row: row['09_day_list'][1:] +  row['10_day_list'][1:]
                                           + row['11_day_list'][1:] +  row['12_day_list'][1:], axis=1)
    return df_all

def get_weekday_seq_entropy(df_all, m):

    df_all['09_weekday_entropy' + '_' + str(m)]    = df_all['09_weekday_seq'].apply(lambda x: ApEn(x, m=m, r=np.mean(x)*0.2))
    df_all['10_weekday_entropy' + '_' + str(m)]    = df_all['10_weekday_seq'].apply(lambda x: ApEn(x, m=m, r=np.mean(x)*0.2))
    df_all['11_weekday_entropy' + '_' + str(m)]    = df_all['11_weekday_seq'].apply(lambda x: ApEn(x, m=m, r=np.mean(x)*0.2))
    df_all['12_weekday_entropy' + '_' + str(m)]    = df_all['12_weekday_seq'].apply(lambda x: ApEn(x, m=m, r=np.mean(x)*0.2))
    df_all['total_weekday_entropy' + '_' + str(m)] = df_all['total_weekday_seq'].apply(lambda x: ApEn(x, m=m, r=np.mean(x)*0.2))
    return df_all


def get_weekend_seq_entropy(df_all, m):

    df_all['09_weekend_entropy' + '_' + str(m)]    = df_all['09_weekend_seq'].apply(lambda x: ApEn(x, m=m, r=np.mean(x)*0.2))
    df_all['10_weekend_entropy' + '_' + str(m)]    = df_all['10_weekend_seq'].apply(lambda x: ApEn(x, m=m, r=np.mean(x)*0.2))
    df_all['11_weekend_entropy' + '_' + str(m)]    = df_all['11_weekend_seq'].apply(lambda x: ApEn(x, m=m, r=np.mean(x)*0.2))
    df_all['12_weekend_entropy' + '_' + str(m)]    = df_all['12_weekend_seq'].apply(lambda x: ApEn(x, m=m, r=np.mean(x)*0.2))
    df_all['total_weekend_entropy' + '_' + str(m)] = df_all['total_weekend_seq'].apply(lambda x: ApEn(x, m=m, r=np.mean(x)*0.2))
    return df_all

def add_at_risk_label(df_all):
    at_rsk_label            = pd.read_csv('Std_list_atRist_2016_se1.csv')
    at_rsk_label['at_risk'] = at_rsk_label['CUM_GPA'].apply(lambda x: '1' if x <= 2.0 else '0')
    at_rsk_label.columns    = ['De-id', 'CUM_GPA', 'at_risk']
    df_all                  = pd.merge(df_all, at_rsk_label, on='De-id', how='left')
    df_all['at_risk']       = df_all['at_risk'].fillna('0')
    return df_all


def get_weekday(day_list):
    weekday_list = []
    for i in range(len(day_list)):
        if (i + 4) % 7 > 0 and (i + 4) % 7 <= 5: # 4 indicates 2016.9.1 is Thursday
            weekday_list.append(day_list[i])
    return weekday_list

def get_weekend(day_list):
    weekend_list = []
    for i in range(len(day_list)):
        if (i + 4) % 7 > 0 and (i + 4) % 7 <= 5: # 4 indicates 2016.9.1 is Thursday
            pass
        else:
            weekend_list.append(day_list[i])
    return weekend_list

def int_handle_cnt(internel_handle_list, df_int_handle, name):
    df_temp = df_int_handle[df_int_handle['internal_handle'].isin(internel_handle_list)]
    df_temp = df_temp.groupby(['De-id']).count().reset_index('De-id')
    df_temp.columns = ['De-id', PRE_FIX + name]
    return df_temp

def extract_one_month(df, PRE_FIX):
    df_t = df[(df['event_type']=='PAGE_ACCESS') |
              (df['event_type']=='COURSE_ACCESS') |
              (df['event_type']=='LOGIN_ATTEMPT') |
              (df['event_type']=='SESSION_TIMEOUT') |
              (df['event_type']=='LOGOUT')]
    df_t = df_t[['De-id', 'event_type', 'course_id', 'internal_handle', 'timestamp']]

    df_evt = df_t[['De-id', 'event_type']]
    df_login = df_evt[df_evt['event_type'] == 'LOGIN_ATTEMPT'].groupby(['De-id']).count().reset_index('De-id')
    df_login.columns = ['De-id', PRE_FIX + 'LOGIN_ATTEMPT']

    df_se_out = df_evt[df_evt['event_type'] == 'SESSION_TIMEOUT'].groupby(['De-id']).count().reset_index('De-id')
    df_se_out.columns = ['De-id', PRE_FIX + 'SESSION_TIMEOUT']

    df_logout = df_evt[df_evt['event_type'] == 'LOGOUT'].groupby(['De-id']).count().reset_index('De-id')
    df_logout.columns = ['De-id', PRE_FIX + 'LOGOUT']

    df_all = df_login
    df_all = pd.merge(df_all, df_se_out, on='De-id', how='left')
    df_all = pd.merge(df_all, df_logout, on='De-id', how='left')

    df_int_handle = df_t[['De-id', 'internal_handle']]

    group_list        = ['groups', 'cp_group_create_self_groupmem', 'group_file', 'group_file', 'group_forum', 'groups_sign_up', 'agroup', 'group_blogs','group_task_create', 'group_task_view','cp_group_edit_self_groupmem','group_file_add', 'group_email', 'cp_groups', 'cp_groups_settings','edit_group_blog_entry', 'db_forum_collection_group', 'group_tasks', 'group_journal','group_virtual_classroom', 'add_group_journal_entry','email_all_groups', 'edit_group_journal_entry', 'email_select_groups', 'add_group_blog_entry']
    db_list           = ['discussion_board_entry', 'db_thread_list_entry', 'discussion_board', 'db_thread_list','db_collection', 'db_collection_group', 'db_collection_entry', 'db_thread_list_group']
    myinfo_list       = ['my_inst_personal_info', 'my_inst_personal_settings','my_inst_personal_edit', 'my_inst_myplaces_settings','my_tasks', 'my_task_create', 'my_email_courses','my_task_view', 'my_announcements']
    course_list       = ['course_tools_area', 'course_task_view', 'enroll_course', 'classic_course_catalog']
    journal_list      = ['journal', 'journal_view', 'view_draft_journal_entry',  'add_journal_entry', 'edit_journal_entry']
    email_list        = ['send_email', 'email_all_instructors', 'email_all_students', 'email_select_students','email_all_users',  'email_select_groups','email_all_groups']
    staff_list        = ['staff_information', 'cp_staff_information']
    annoucements_list = ['my_announcements', 'announcements_entry', 'announcements', 'cp_announcements']
    content_list      = ['content', 'cp_content']
    grade_list        = ['check_grade']

    df_group        = int_handle_cnt(group_list, df_int_handle, 'group')
    df_db           = int_handle_cnt(db_list, df_int_handle, 'db')
    df_myinfo       = int_handle_cnt(myinfo_list, df_int_handle, 'myinfo')
    df_course       = int_handle_cnt(course_list, df_int_handle, 'course')
    df_journal      = int_handle_cnt(journal_list, df_int_handle, 'journal')
    df_email        = int_handle_cnt(email_list, df_int_handle, 'email')
    df_staff        = int_handle_cnt(staff_list, df_int_handle, 'staff')
    df_annoucements = int_handle_cnt(annoucements_list, df_int_handle, 'annoucements')
    df_content      = int_handle_cnt(content_list, df_int_handle, 'content')
    df_grade        = int_handle_cnt(grade_list, df_int_handle, 'grade')

    dfs = [df_group, df_db, df_myinfo, df_course, df_journal, df_email, df_staff, df_annoucements, df_content, df_grade]

    for df in dfs:
        df_all = pd.merge(df_all, df, on='De-id', how='left')   

    df_all = df_all.rename(columns={'De-id':'MASKED_STUDENT_ID'})
    return df_all


if __name__ == '__main__':
    df_all = get_all_seq(data_path_list, 'COURSE_ACCESS')
    # df_all = get_seq_entropy(df_all, 7)
    
    df_all['09_weekday_seq'] = df_all['09_day_list'].apply(get_weekday)
    df_all['09_weekend_seq'] = df_all['09_day_list'].apply(get_weekday)

    df_all['10_weekday_seq'] = df_all['10_day_list'].apply(get_weekday)
    df_all['10_weekend_seq'] = df_all['10_day_list'].apply(get_weekday)

    df_all['11_weekday_seq'] = df_all['11_day_list'].apply(get_weekday)
    df_all['11_weekend_seq'] = df_all['11_day_list'].apply(get_weekday)

    df_all['12_weekday_seq'] = df_all['12_day_list'].apply(get_weekday)
    df_all['12_weekend_seq'] = df_all['12_day_list'].apply(get_weekday)

    df_all['total_weekday_seq'] = df_all['total_list'].apply(get_weekday)
    df_all['total_weekend_seq'] = df_all['total_list'].apply(get_weekday)

    df_all = get_weekday_seq_entropy(df_all, 5)
    df_all = get_weekend_seq_entropy(df_all, 2)
    df_all = add_at_risk_label(df_all)

    n_list = list(df_all.columns)
    pattern = re.compile('.*_entropy_.*')
    entropy_list = ['De-id']
    for i in n_list:
        if pattern.match(i):
            entropy_list.append(i)

    df_all_entropy = df_all[entropy_list]
    df_all_entropy = df_all_entropy.rename(columns={'De-id':'MASKED_STUDENT_ID'})
    
    ''' 
        Till this get all seq entropy features
    
    '''
    
    lib_se1 = pd.read_csv('Std_Lib_features_2016_se1.csv')
    his_2015_se1 = pd.read_csv('Std_list_atRist_2015_se1.csv')
    his_2015_se2 = pd.read_csv('Std_list_atRist_2015_se2.csv')
    his_2015_se1.columns = ['MASKED_STUDENT_ID', '2015_se1_CUM_GPA']
    his_2015_se2.columns = ['MASKED_STUDENT_ID', '2015_se2_CUM_GPA']

    his_lib = pd.merge(lib_se1, his_2015_se1, on='MASKED_STUDENT_ID', how='left').fillna(0)
    his_lib = pd.merge(his_lib, his_2015_se2, on='MASKED_STUDENT_ID', how='left').fillna(0)

    df_se1 = pd.merge(df_all_entropy, his_lib, on='MASKED_STUDENT_ID', how='left').fillna(0)
    
    ''' 
        Add historical grades for one year
    
    '''
    
    
    # lib_se1 = pd.read_csv('Std_Lib_features_2016_se1.csv')
    # df_se1  = lib_se1

    df = pd.read_csv('DR0008_activity_accumulator_2016_09.csv', sep='\t')
    df['weekday'] = pd.to_datetime(df['timestamp']).dt.dayofweek
    df['is_weekday'] = df['weekday'].apply(lambda x: 1 if x <=5 else 0)
    df_weekday = df[df['is_weekday'] == 1]
    df_weekend = df[df['is_weekday'] == 0]

    PRE_FIX = '09_weekday_'
    df_weekday_one_month = extract_one_month(df_weekday, PRE_FIX)
    PRE_FIX = '09_weekend_'
    df_weekend_one_month = extract_one_month(df_weekend, PRE_FIX)
    df_se1 = pd.merge(df_se1, df_weekday_one_month, on = ['MASKED_STUDENT_ID'], how='left').fillna(0)
    df_se1 = pd.merge(df_se1, df_weekend_one_month, on = ['MASKED_STUDENT_ID'], how='left').fillna(0)
    del df
    del df_weekday_one_month
    del df_weekend_one_month

    df = pd.read_csv('DR0008_activity_accumulator_2016-10.csv', sep='\t')
    df['weekday'] = pd.to_datetime(df['timestamp']).dt.dayofweek
    df['is_weekday'] = df['weekday'].apply(lambda x: 1 if x <=5 else 0)
    df_weekday = df[df['is_weekday'] == 1]
    df_weekend = df[df['is_weekday'] == 0]

    PRE_FIX = '10_weekday_'
    df_weekday_one_month = extract_one_month(df_weekday, PRE_FIX)
    PRE_FIX = '10_weekend_'
    df_weekend_one_month = extract_one_month(df_weekend, PRE_FIX)
    df_se1 = pd.merge(df_se1, df_weekday_one_month, on = ['MASKED_STUDENT_ID'], how='left').fillna(0)
    df_se1 = pd.merge(df_se1, df_weekend_one_month, on = ['MASKED_STUDENT_ID'], how='left').fillna(0)
    del df
    del df_weekday_one_month
    del df_weekend_one_month

    df = pd.read_csv('DR0008_activity_accumulator_2016-11.csv', sep='\t')
    df['weekday'] = pd.to_datetime(df['timestamp']).dt.dayofweek
    df['is_weekday'] = df['weekday'].apply(lambda x: 1 if x <=5 else 0)
    df_weekday = df[df['is_weekday'] == 1]
    df_weekend = df[df['is_weekday'] == 0]

    PRE_FIX = '11_weekday_'
    df_weekday_one_month = extract_one_month(df_weekday, PRE_FIX)
    PRE_FIX = '11_weekend_'
    df_weekend_one_month = extract_one_month(df_weekend, PRE_FIX)
    df_se1 = pd.merge(df_se1, df_weekday_one_month, on = ['MASKED_STUDENT_ID'], how='left').fillna(0)
    df_se1 = pd.merge(df_se1, df_weekend_one_month, on = ['MASKED_STUDENT_ID'], how='left').fillna(0)
    del df
    del df_weekday_one_month
    del df_weekend_one_month

    df = pd.read_csv('DR0008_activity_accumulator_2016-12.csv', sep='\t')
    df['weekday'] = pd.to_datetime(df['timestamp']).dt.dayofweek
    df['is_weekday'] = df['weekday'].apply(lambda x: 1 if x <=5 else 0)
    df_weekday = df[df['is_weekday'] == 1]
    df_weekend = df[df['is_weekday'] == 0]

    PRE_FIX = '12_weekday_'
    df_weekday_one_month = extract_one_month(df_weekday, PRE_FIX)
    PRE_FIX = '12_weekend_'
    df_weekend_one_month = extract_one_month(df_weekend, PRE_FIX)
    df_se1 = pd.merge(df_se1, df_weekday_one_month, on = ['MASKED_STUDENT_ID'], how='left').fillna(0)
    df_se1 = pd.merge(df_se1, df_weekend_one_month, on = ['MASKED_STUDENT_ID'], how='left').fillna(0)
    del df
    del df_weekday_one_month
    del df_weekend_one_month

    '''
        Till this got LMS week statistical features
    
    '''

    # merge feature
    df_se1 = pd.merge(df_se1, df_all_entropy, on='MASKED_STUDENT_ID', how='left').fillna(0)
    df_se1_features = df_se1[[i for i in df_se1.columns if i != 'label_atRist' and i != 'MASKED_STUDENT_ID']]
    df_se1_labels = df_se1['label_atRist']

    # classification
    X_train, X_test, y_train, y_test = train_test_split(df_se1_features, df_se1_labels, test_size = 0.2, stratify=df_se1_labels)

    brf = BalancedRandomForestClassifier(n_estimators=300, criterion = 'gini', random_state=0)
    brf.fit(X_train, y_train) 
    y_pred = brf.predict(X_test)
    imp_feature = brf.feature_importances_
    print(confusion_matrix(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))
    print(balanced_accuracy_score(y_test, y_pred))

    '''
        2019.5.30
        1. 加了lms统计特征(weekday and weekend)和seq统计特征（weekday and weekend)
        2. 划分训练集采用同分布
        Best acc now: 0.7309003914745542
        
        Next Step:
        1. 加入历史成绩特征
        2. 观察具体 weekday 和 weekend 两类数据分布具体有何不同
        3. 测试用半学期行为数据early predict
        
        
        2019.6.13
        1. 加入历史成绩特征
        Best acc now: 0.7534047834691893
        
        Next Step:
        1. 观察具体 weekday 和 weekend 两类数据分布具体有何不同
        2. 测试用半学期行为数据early predict
    '''





