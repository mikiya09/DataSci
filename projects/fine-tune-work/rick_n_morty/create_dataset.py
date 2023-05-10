import pandas as pd 

# Load csv 
# df = pd.read_csv("old-newspaper.tsv", delimiter="\t")
# convert tsv to csv 
# df.to_csv("oldNewspaper.csv", index=False)


# load new csv 
df = pd.read_csv("RickAndMortyScripts.csv")
print(df.columns)
print(len(df))
# print(df["Language"].unique())
# check how many rows for selected language 

# English = df[df["Language"] == "English"]
# English.to_csv("English.csv", index=False)
# Chinese = df[df["Language"] == "Chinese (Traditional)"]
# Chinese.to_csv("Chinese.csv", index=False)
# Japanese = df[df["Language"] == "Japanese"]
# Japanese.to_csv("Japanese.csv", index=False)
# German = df[df["Language"] == "German"]
# German.to_csv("German.csv", index=False)
# French = df[df["Language"] == "French"]
# French.to_csv("French.csv", index=False)

# ========== convert .csv to .txt file =============

# df = pd.read_csv("./dataset/Chinese.csv")
# # # print(df.head())
# # # print(len(df))

df_short = df.sample(n=1250, random_state=1)
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(df_short, test_size=0.2, random_state=1)


# text_full = '\n'.join(df['Text'])
text_train = '\n'.join(train_set['line'])
text_test = '\n'.join(test_set["line"])
# Write the text data to a single TXT file 
# with open('Chinese_full.txt', 'w') as file:
#     file.write(text_full)

with open('rick_n_morty_train.txt', 'w') as file:
    file.write(text_train)

with open('rick_n_morty_test.txt', 'w') as file:
    file.write(text_test)

# =========== Fine Tune ============ 
