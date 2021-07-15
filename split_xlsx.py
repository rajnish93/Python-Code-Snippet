# Import library
import pandas as pd
# Reading File
xlsx = pd.ExcelFile('./Dataset/Gold_Prices.xlsx')
# Iterate over sheet_names
for name in xlsx.sheet_names:
    # Read Data from each sheet
    df = pd.read_excel(xlsx, name)
    # Save to csv file with file name as sheet_name
    # in output directory
    dir='output'
    df.to_csv(dir +'/'+name+'.csv', index=False)