# Predicting armed conflict using protest data - data / model management / training

from viewser import Queryset, Column
import sqlalchemy as sa
from ingester3.config import source_db_path
import pandas as pd
import numpy as np
import geopandas as gpd



def SummarizeTable(dfname,df):
    print(f"{dfname}: A dataset with {len(df.columns)} columns, with "
      f"data between t = {min(df.index.get_level_values(0))} "
      f"and {max(df.index.get_level_values(0))}; "
      f"{len(np.unique(df.index.get_level_values(1)))} units."
     )
    
def FetchTable(Queryset, name):
    df = Queryset.fetch()#.astype(float)
    df.name = name
    SummarizeTable(name,df)
    Data = {
            'Name': name,
            'df': df
        }
    return(Data)

def FetchData(run_id):
    print('Fetching data using querysets; returns as list of dictionaries containing datasets')
    Datasets = []
    if run_id == 'baseline_incidence':
        # Baseline models
        Datasets.append(FetchTable((Queryset("protest_paper_old_baseline_incidence", "priogrid_month")),'baseline_simple'))
        Datasets.append(FetchTable((Queryset("protest_paper_extended_baseline_incidence", "priogrid_month")),'baseline'))
        
        return(Datasets)
    
    if run_id == 'small_set_models_2909_incidence':
        # Baseline models
        Datasets.append(FetchTable((Queryset("protest_paper_old_baseline_incidence", "priogrid_month")),'baseline_simple'))
        Datasets.append(FetchTable((Queryset("protest_paper_extended_baseline_incidence", "priogrid_month")),'baseline'))
        Datasets.append(FetchTable((Queryset("protest_paper_pr_naive_bl", "priogrid_month")),'pr_naive_bl'))
        Datasets.append(FetchTable((Queryset("protest_paper_pr_dynamic_local_bl", "priogrid_month")),'pr_dynamic_loc_bl'))
        Datasets.append(FetchTable((Queryset("protest_paper_pr_dynamic_national_bl", "priogrid_month")),'pr_dynamic_nat_bl'))
        
        return(Datasets)
    
    if run_id =='main_models_0810_2022_incidence_01':
        # Baseline models
        Datasets.append(FetchTable((Queryset("protest_paper_old_baseline_incidence", "priogrid_month")),'baseline_simple'))
        Datasets.append(FetchTable((Queryset("protest_paper_pr_naive_bl", "priogrid_month")),'pr_naive_bl'))
        Datasets.append(FetchTable((Queryset("protest_paper_pr_dynamic_local_bl", "priogrid_month")),'pr_dynamic_loc_bl'))
        Datasets.append(FetchTable((Queryset("protest_paper_pr_dynamic_national_bl", "priogrid_month")),'pr_dynamic_nat_bl'))
        Datasets.append(FetchTable((Queryset("protest_paper_pr_elecdemo_bl", "priogrid_month")),'pr_elecdemo_bl'))
        Datasets.append(FetchTable((Queryset("protest_paper_pr_civlib_bl", "priogrid_month")),'pr_civlib_bl'))
        Datasets.append(FetchTable((Queryset("protest_paper_pr_elect_bl", "priogrid_month")),'pr_elect_bl'))
        
        return(Datasets)
    
    if run_id =='main_models_0810_2022_incidence_02':
        # Baseline models
        Datasets.append(FetchTable((Queryset("protest_paper_pr_econ_national_bl", "priogrid_month")),'pr_econ_nat_bl'))
        Datasets.append(FetchTable((Queryset("protest_paper_pr_econ_full_bl", "priogrid_month")),'pr_econ_full_bl'))
        Datasets.append(FetchTable((Queryset("protest_paper_pr_devi_bl_01", "priogrid_month")),'pr_devi_bl'))
        Datasets.append(FetchTable((Queryset("protest_paper_pr_devi_bl_02", "priogrid_month")),'pr_devi_bl'))
        return(Datasets)
        
    if run_id == 'protest_paper_0109_2022_incidence_01':
        # Baseline models
        Datasets.append(FetchTable((Queryset("protest_paper_old_baseline_incidence", "priogrid_month")),'baseline_simple'))
        # Baseline + Economic development models, country level and subnational level
        Datasets.append(FetchTable((Queryset("protest_paper_econ_national_bl", "priogrid_month")),'econ_nat_bl'))
        Datasets.append(FetchTable((Queryset("protest_paper_econ_full_bl", "priogrid_month")),'econ_full_bl'))
        # Baseline + Political instiutions models (I, II, III, IV)
        Datasets.append(FetchTable((Queryset("protest_paper_inst_elecdemo_bl", "priogrid_month")),'inst_elecdemo_bl'))
        Datasets.append(FetchTable((Queryset("protest_paper_inst_civlib_bl", "priogrid_month")),'inst_civlib_bl'))
        Datasets.append(FetchTable((Queryset("protest_paper_inst_elect_bl", "priogrid_month")),'inst_elect_bl'))
        Datasets.append(FetchTable((Queryset("protest_paper_inst_devi_bl", "priogrid_month")),'inst_devi_bl'))
        # Baseline + Political instiutions models (III, IV) + economic development models
        Datasets.append(FetchTable((Queryset("protest_paper_elect_econ_national_bl", "priogrid_month")),'elect_econ_nat_bl'))
        Datasets.append(FetchTable((Queryset("protest_paper_elect_econ_full_bl", "priogrid_month")),'elect_econ_full_bl'))
        Datasets.append(FetchTable((Queryset("protest_paper_devi_econ_national_bl", "priogrid_month")),'devi_econ_nat_bl'))
        Datasets.append(FetchTable((Queryset("protest_paper_devi_econ_full_bl", "priogrid_month")),'devi_econ_full_bl'))
        # Baseline + Protest models
        Datasets.append(FetchTable((Queryset("protest_paper_pr_naive_bl", "priogrid_month")),'pr_naive_bl'))
        Datasets.append(FetchTable((Queryset("protest_paper_pr_dynamic_local_bl", "priogrid_month")),'pr_dynamic_loc_bl'))
        Datasets.append(FetchTable((Queryset("protest_paper_pr_dynamic_national_bl", "priogrid_month")),'pr_dynamic_nat_bl'))
         # Baseline + Full protest model + political instiutions models (I, II, III, IV)
        Datasets.append(FetchTable((Queryset("protest_paper_pr_elecdemo_bl", "priogrid_month")),'pr_elecdemo_bl'))
        Datasets.append(FetchTable((Queryset("protest_paper_pr_civlib_bl", "priogrid_month")),'pr_civlib_bl'))
        Datasets.append(FetchTable((Queryset("protest_paper_pr_elect_bl", "priogrid_month")),'pr_elect_bl'))
        return(Datasets)
    
    if run_id == 'protest_paper_0109_2022_incidence_02':
        # Baseline + Full protest model + econmic development models
        Datasets.append(FetchTable((Queryset("protest_paper_pr_econ_national_bl", "priogrid_month")),'pr_econ_nat_bl'))
        Datasets.append(FetchTable((Queryset("protest_paper_pr_econ_full_bl", "priogrid_month")),'pr_econ_full_bl'))
        # Baseline + Full protest model + political instiutions (III, IV) + econmic development models
        Datasets.append(FetchTable((Queryset("protest_paper_pr_elect_econ_national_bl", "priogrid_month")),'pr_elect_econ_nat_bl'))
        Datasets.append(FetchTable((Queryset("protest_paper_pr_elect_econ_full_bl", "priogrid_month")),'pr_elect_econ_full_bl'))
        
        return(Datasets)
    
    if run_id == 'protest_paper_0109_2022_incidence_03_01':
        # Tables with more than 50 variables.
        Datasets.append(FetchTable((Queryset("protest_paper_pr_devi_bl_01", "priogrid_month")),'pr_devi_bl'))
        Datasets.append(FetchTable((Queryset("protest_paper_pr_devi_econ_national_bl_01", "priogrid_month")),'pr_devi_econ_nat_bl'))
        Datasets.append(FetchTable((Queryset("protest_paper_pr_devi_econ_full_bl_01", "priogrid_month")),'pr_devi_econ_full_bl'))
        
        return(Datasets)
    
    if run_id == 'protest_paper_0109_2022_incidence_03_02':
        # Tables with more than 50 variables.
        Datasets.append(FetchTable((Queryset("protest_paper_pr_devi_bl_02", "priogrid_month")),'pr_devi_bl'))
        Datasets.append(FetchTable((Queryset("protest_paper_pr_devi_econ_national_bl_02", "priogrid_month")),'pr_devi_econ_nat_bl'))
        Datasets.append(FetchTable((Queryset("protest_paper_pr_devi_econ_full_bl_02", "priogrid_month")),'pr_devi_econ_full_bl'))
        
        return(Datasets)
    
def MergeQueries(df1,df2, name):
    df = pd.concat([df1,df2],axis=1)
    df1.name = name
    Data = {
            'Name': name,
            'df': df
        }
    return(Data)
    
def getDuplicateColumns(df):
    
    if df.columns.duplicated().any() == True:
        print('Duplicates detected')
        
        print('Remove duplicates')
        df = df.loc[:,~df.columns.duplicated()]
        
    else:
        ('No duplicates')
  
    return df

def data_integrity_check(dataset, depvar):
    
    # Check if dependent variable is in dataset. 
    if depvar not in dataset['df'].columns:
        print(depvar, 'not found in', dataset['Name'])
        return
    
    # Make sure that dependent variable is in first column.
    if dataset['df'].columns[0] != depvar:
        print('Reordering columns in model', dataset['Name'])
        depvar_column = dataset['df'].pop(depvar)
        dataset['df'].insert(0, depvar, depvar_column)
        
    # Check for duplicates in columns.
    print('Drop duplicates', dataset['Name'])
    dataset['df'] = getDuplicateColumns(dataset['df'])
        
    # Check for Nas.
    print('Checking for NaN/Null')
    for column in dataset['df'].columns:
        if dataset['df'][column].isna().sum() != 0:
            print('WARNING - NaN/Null data detected in', dataset['Name'], 'column', column)
            
    
    

    return

def fetch_africa_ids():
    qs = (Queryset("protest_paper_in_africa_only_", "priogrid_month")

        # target variable
        .with_column(Column("in_africa", from_table = "priogrid_month", from_column = "in_africa")
            )
         )
      
    df_pg = qs.publish().fetch()
    
    return df_pg

def crop_africa(dataset,df_pg):
    
    dfs=[dataset['df'], df_pg]
    merged_df = pd.concat(dfs,axis=1)#.astype(float)
    dataset['df'] = merged_df[merged_df.in_africa == True]
    
    dataset['df'] = dataset['df'].drop(labels=['in_africa'], axis=1)
    
    SummarizeTable(dataset['Name'],dataset['df'])
    
    return

def crop_months(df,mon_min,mon_max):
    
    df_ret = df.copy()
    df_ret = df.loc[mon_min:mon_max,:]
    
    return df_ret

def reindex_df(df):
    new_index = df.index.remove_unused_levels()
    df_ret = df.set_index(new_index)
    
    return df_ret

def fetch_gdf():
    engine = sa.create_engine(source_db_path)
    gdf = gpd.GeoDataFrame.from_postgis(
    "SELECT id as pg_id, in_africa, in_me, geom FROM prod.priogrid", 
    engine, 
    geom_col='geom'
    )
    
    # Set crs, rename column
    gdf_master = gdf.to_crs(4326)
    gdf_master = pd.DataFrame(gdf_master.rename(columns={'geom':'geometry','pg_id':'priogrid_gid'}).set_index('priogrid_gid')['geometry'])
    
    # Get index with month id
    df_index = fetch_africa_ids()
    df_index = df_index[df_index.in_africa == True]
    
    df_index = crop_months(df_index,200,520)
    df_index = reindex_df(df_index)
    
    # Join
    df_geom = df_index.join(gdf_master,how='left')
    df_geom = df_geom.drop(columns=['in_africa'])
    return df_geom








