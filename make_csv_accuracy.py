import os
import argparse
import pandas as pd 
import glob

ap=argparse.ArgumentParser()
ap.add_argument('-c','--csv',type=str,default='',help='csv file folder path')
args=vars(ap.parse_args())

c=args['csv']
Time_stamp_name = c+".csv"
# print(Time_stamp_name)

csv_file_list = glob.glob(c+'/*.csv')
csv_file_list.sort()
print(csv_file_list)

data =  []
th_value = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for th in th_value:
    for csv in csv_file_list:
        m = (csv.split('.csv')[0]+'.h5').split('/')[-1]
        
        df = pd.read_csv(csv)
        # print(df)

        # total_real_images_count = df.loc[:,'GT']
        # print(total_real_images_count)
        try:

            total_real_images_count = df['GT'].value_counts()['real']
            total_fake_images_count = df['GT'].value_counts()['fake']

            # print(df.dtypes)
            df1 = df.loc[df['pred_score']>=th]
            df2 = df1.loc[df1['GT']=='real']
            try:
                total_pred_real_images_count = df2['GT'].value_counts()['real']
                # print(total_pred_real_images_count)
            except:
                total_pred_real_images_count = 0


            df1 = df.loc[df['pred_score']<(th)]
            df2 = df1.loc[df1['GT']=='fake']
            try : 
                total_pred_fake_images_count = df2['GT'].value_counts()['fake']
                # print(total_pred_fake_images_count)
            except:
                total_pred_fake_images_count = 0

            real_accuracy = (total_pred_real_images_count/total_real_images_count)*100
            fake_accuracy = (total_pred_fake_images_count/total_fake_images_count)*100

            data.append((m,th,real_accuracy,fake_accuracy))
            csv_df = pd.DataFrame(data, columns=['model_name','th_value', 'real_acc', 'fake_acc'])
        except:
            print("key_value_error")
print(csv_df)
csv_df.to_csv(Time_stamp_name,index=False)
