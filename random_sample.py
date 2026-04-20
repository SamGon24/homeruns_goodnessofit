import pandas as pd

df = pd.read_csv('sorted_dataset.csv')

sample = df.sample(n=200, random_state=42).reset_index(drop=True)

sample.to_csv('sample_200.csv', index=False)

print(f"Saved {len(sample)} rows to sample_200.csv")
print(sample[['Player', 'Year', 'HR']].to_string())