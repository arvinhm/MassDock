import pandas as pd

# 1. load
df = pd.read_csv('dock_boxes.csv')

# 2. remove the ".pdb" extension
df['filename'] = df['filename'].str.replace(r'\.pdb$', '', regex=True)

# 3. split into [proteinID, chainLigand, chainReceptor]
parts = df['filename'].str.split('_', expand=True)
df['proteinID']     = parts[0]
df['chainReceptor'] = parts[2]

# 4. reconstruct filename WITHOUT the ligand chain
df['filename'] = df['proteinID'] + '_' + df['chainReceptor']

# 5. dedupe by proteinID, keeping first hit
df = df.drop_duplicates(subset='proteinID', keep='first')

# 6. clean up helper columns and save
df = df.drop(columns=['proteinID', 'chainReceptor'])
df.to_csv('dock_boxes_filtered.csv', index=False)
