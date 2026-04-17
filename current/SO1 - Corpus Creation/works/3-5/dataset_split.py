import pandas as pd
from sklearn.model_selection import train_test_split

# Load your master corpus
df = pd.read_csv("master_parallel_corpus__COMBINED.csv")

# Ensure required columns exist
assert all(col in df.columns for col in ["domain", "sentence_type"])

# Stratified split
train, temp = train_test_split(
    df,
    test_size=0.3,
    stratify=df["domain"],
    random_state=42
)

dev, test = train_test_split(
    temp,
    test_size=0.5,
    stratify=temp["domain"],
    random_state=42
)

# Save
train.to_csv("train.csv", index=False)
dev.to_csv("dev.csv", index=False)
test.to_csv("test.csv", index=False)

print("Split complete:")
print(len(train), len(dev), len(test))