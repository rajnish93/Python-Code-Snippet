# Concatenate training and test Dataset
data_train = data_train[list(data_test)]
all_data = pd.concat((data_train, data_test))
print(data_train.shape, data_test.shape, all_data.shape)
# Apply Dummies
all_data = pd.concat([all_data, pd.get_dummies(
    all_data['dose'], prefix='dose', dtype=float)], axis=1)
all_data = pd.concat([all_data, pd.get_dummies(
    all_data['time'], prefix='time', dtype=float)], axis=1)
all_data = pd.concat([all_data, pd.get_dummies(
    all_data['category'], prefix='category', dtype=float)], axis=1)
all_data = all_data.drop(['dose', 'time', 'category'], axis=1)
# Split
train = all_data[:len(data_train)]
test = all_data[len(data_train):]
print(train.shape, test.shape)
# another way
df_train, df_test = train_test_split(
    df, test_size=0.1, random_state=RANDOM_SEED, stratify=df.label.values)
df_val, df_test = train_test_split(
    df_test, test_size=0.5, random_state=RANDOM_SEED, stratify=df_test.label.values)