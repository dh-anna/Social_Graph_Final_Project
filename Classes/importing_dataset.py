import pandas as pd
def import_dataset(file_path):

    df = pd.read_csv(file_path, low_memory=False)

    return df

def saving_df_to_pickle(df, name):
    df.to_pickle(f"{name}.pkl")

def loading_df_from_pickle(name):
    df = pd.read_pickle(f"{name}.pkl")
    return df
