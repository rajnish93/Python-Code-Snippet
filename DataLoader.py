def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = RKDataset(
        reviews=df.description.to_numpy(),
        targets=df.label.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4
    )
train_data_loader = create_data_loader(
    df_train, tokenizer, MAX_LEN, BATCH_SIZE)
val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)
# Another way
train_dataset = RKDataset(reviews=df_train.description.values,
                          targets=df_train.label.values
                          )
train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=TRAIN_BATCH_SIZE,
                                                num_workers=4
                                                )
valid_dataset = RKDataset(reviews=df_val.description.values,
                          targets=df_val.label.values
                          )
val_data_loader = torch.utils.data.DataLoader(valid_dataset,
                                              batch_size=VALID_BATCH_SIZE,
                                              num_workers=1
                                              )
test_dataset = RKDataset(reviews=df_test.description.values,
                         targets=df_test.label.values
                         )
test_data_loader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=VALID_BATCH_SIZE,
                                               num_workers=1
                                               )
data = next(iter(train_data_loader))
data.keys()
print(data['input_ids'].shape)
print(data['attention_mask'].shape)
print(data['targets'].shape)