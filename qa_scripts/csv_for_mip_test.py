import pandas as pd

df = pd.read_csv('csv/windowed_mip_result.csv')

df = df.loc[:, ['subj', 'cta_windowed', 'mip_windowed', 'auto_windowed_mip']]

df['auto_windowed_mip']=df['cta_windowed'].apply(lambda x: x.replace('.nii.gz', '_mip_{}.nii.gz'.format('d'+str(2)+'s'+str(120)+'t'+str(40))))

df.to_csv('csv/mip_test.csv')