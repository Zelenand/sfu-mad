encoded_col_names = {}
for i in dataframe[cat_atributes[0]].unique():
    for index, row in dataframe.iterrows():
        if row[cat_atributes[0]] == i:
            for j in encoder_df.columns:
                if row[j] == 1.0:
                    encoded_col_names[str(j)] = i
            break
for i in encoded_col_names:
    dataframe.rename(columns = {int(i): encoded_col_names[i]}, inplace = True)