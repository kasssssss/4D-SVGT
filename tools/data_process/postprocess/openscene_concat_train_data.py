import pandas as pd
import glob
import os

directory_path = 'data_annotation/meta/moge_v2_large_correct_focal/moge_pred_filter_parquet' 
output_filename = os.path.join(directory_path, 'openscene_total_train.parquet')

parquet_files = glob.glob(os.path.join(directory_path, 'openscene_trainval_slice_*.parquet'))

df_list = []

for file in parquet_files:
    df = pd.read_parquet(file)
    df_list.append(df)
    print(f"已成功读取: {os.path.basename(file)}")

total_df = pd.concat(df_list, ignore_index=True)
print("数据合并完成。")


total_df.to_parquet(output_filename, engine='pyarrow', compression='zstd', index=False)
print(f"结果在 '{output_filename}'...")
