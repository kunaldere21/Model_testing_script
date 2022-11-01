import os
import argparse
import pandas as pd 

ap=argparse.ArgumentParser()
ap.add_argument('-c','--csv',type=str,default='',help='csv file folder path')
args=vars(ap.parse_args())

c=args['csv']
csv_file_list = os.listdir(c)

txt_path = os.path.join(c,'my_accurracy.txt')

for csv in csv_file_list:
    m = csv.split('.csv')[0]+'.h5'
    
    df = pd.read_csv(os.path.join(c,csv))

    total_real_images_count = df['GT'].value_counts()['real']
    total_fake_images_count = df['GT'].value_counts()['fake']
    # print(df.dtypes)
    df1 = df.loc[df['pred_score']>=0.5]
    df2 = df1.loc[df1['GT']=='real']
    try:
        total_pred_real_images_count = df2['GT'].value_counts()['real']
        # print(total_pred_real_images_count)
    except:
        total_pred_real_images_count = 0


    df1 = df.loc[df['pred_score']<0.5]
    df2 = df1.loc[df1['GT']=='fake']
    try : 
        total_pred_fake_images_count = df2['GT'].value_counts()['fake']
        # print(total_pred_fake_images_count)
    except:
        total_pred_fake_images_count = 0

    real_accuracy = (total_pred_real_images_count/total_real_images_count)*100
    fake_accuracy = (total_pred_fake_images_count/total_fake_images_count)*100
    print(real_accuracy,fake_accuracy)
    txt_file = open(txt_path,'a')
    txt_file.write('\n----------------------------------\n')
    txt_file.write('Model name : '+m+'\n')
    txt_file.write('real_accuracy = '+str(real_accuracy)+'\n')
    txt_file.write('fake_accuracy = '+str(fake_accuracy)+'\n')