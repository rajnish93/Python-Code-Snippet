# Create a column
df.loc[:, "kfold"] = -1
df = df.sample(frac=1).reset_index(drop=True)
targets = df.drop(['id'], axis=1).values
mskf = MultilabelStratifiedKFold(n_splits=5)
for fold, (trn, val) in enumerate(mskf.split(X=df, y=targets)):
    # We always take validation can skip trn from above like:(_,val)
    df.loc[val, "kfold"] = fold