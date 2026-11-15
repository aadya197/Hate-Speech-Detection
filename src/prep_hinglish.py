"""Turn a downloaded Hinglish hate/offensive dataset into train/test CSVs (text,label) with HateXplain's labels.

Labels: 0 Hate speech, 1 Normal, 2 Offensive.
Example:
  python -m src.prep_hinglish --csv hing.csv --text_col text --label_col label \
      --map "hate=0,normal=1,non-hate=1,offensive=2,abusive=2"
"""
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

p = argparse.ArgumentParser()
p.add_argument("--csv", required=True)
p.add_argument("--text_col", default="text")
p.add_argument("--label_col", default="label")
p.add_argument("--map", required=True, help="source label (lowercase) = 0/1/2, comma separated")
p.add_argument("--test_size", type=float, default=0.2)
p.add_argument("--out_dir", default="data/processed")
a = p.parse_args()

mp = {k.strip().lower(): int(v) for k, v in (x.split("=") for x in a.map.split(","))}
df = pd.read_csv(a.csv)
df = df[[a.text_col, a.label_col]].dropna()
df["label"] = df[a.label_col].astype(str).str.strip().str.lower().map(mp)
print("unmapped labels dropped:", int(df["label"].isna().sum()))
df = df.dropna(subset=["label"]).rename(columns={a.text_col: "text"})[["text", "label"]]
df["label"] = df["label"].astype(int)
df["text"] = df["text"].astype(str).str.lower().str.replace(r"http\S+|@\w+", " ", regex=True).str.split().str.join(" ")
df = df[df["text"].str.len() > 0].drop_duplicates("text")
tr, te = train_test_split(df, test_size=a.test_size, random_state=42, stratify=df["label"])
tr.to_csv(f"{a.out_dir}/hinglish_train.csv", index=False)
te.to_csv(f"{a.out_dir}/hinglish_test.csv", index=False)
print("train", len(tr), tr["label"].value_counts().to_dict(), "| test", len(te), te["label"].value_counts().to_dict())
