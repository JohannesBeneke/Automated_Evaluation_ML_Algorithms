from os import name
import pandas as pd


abc = pd.DataFrame()

# name='SFGD'
# start_time='13'
# finish_time='15'
# algorithm='RFC'

# dict_1 = {'Name':name,
#         'Start Time':start_time,
#         'Finish Time':finish_time,
#         'Algorithm':algorithm,
# }
# dict_dataframe=pd.DataFrame(dict_1, index=[0])

# abc_new = pd.concat([abc,dict_dataframe], axis=0)

# print(abc)

# print(abc_new)

number_methods = 2
names = ['PCA','SCaler']

number_preprocessing_methods = ['Preprocessing_Method_'+str(count) for count in range(number_methods)]
preprocessing_dict = dict(zip(number_preprocessing_methods, names))

print(preprocessing_dict)

a = **preprocessing_dict

name='SFGD'
start_time='13'
finish_time='15'
algorithm='RFC'

dict_1 = {'Name':name,
        'Start Time':start_time,
        'Finish Time':finish_time,
        'Algorithm':algorithm,
}
dict_dataframe=pd.DataFrame(dict_1, index=[0])
