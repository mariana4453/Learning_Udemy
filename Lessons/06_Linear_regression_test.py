# imports
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

# importing data file
path = './additional_data/data_for_ML.xlsx'
df = pd.read_excel(path)

wb = pd.ExcelFile(path, engine='openpyxl')
excel_sheets = wb.sheet_names

# print(df.info())
# print(len(df))

# (['SUPERSUBJECT_NK', 'CD_STATE', 'D_ORIG', 'D_EXP', 'AMT_ORIG_BAL',
#        'PCT_BASE_RATE', 'CD_INTR_FIX_PERIOD', 'CNT_ALL_SPL_LOAN_REPAYMENTS',
#        'AMT_INSTALLMENT']


# sns.histplot(data=df, x='AMT_ORIG_BAL', kde=True)
# sns.pairplot(data=df)
sns.regplot(data=df, x='AMT_ORIG_BAL', y='AMT_INSTALLMENT')

plt.show()