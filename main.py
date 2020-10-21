def main_fn(fold, params, save_model=False):
    feature_columns = train.drop(['sig_id'], axis=1).columns
    targets_columns = target_scored.drop(["sig_id"], axis=1).columns
    df = train.merge(target_scored, on="sig_id", how="left")
    train_df = df[df.kfold != fold].reset_index(drop=True)
    valid_df = df[df.kfold == fold].reset_index(drop=True)
    xtrain = train_df[feature_columns].to_numpy()
    ytrain = train_df[targets_columns].to_numpy()
    xvalid = valid_df[feature_columns].to_numpy()
    yvalid = valid_df[targets_columns].to_numpy()
    # Dataset
    train_dataset = RKDataset(inputs=xtrain, targets=ytrain)
    valid_dataset = RKDataset(inputs=xvalid, targets=yvalid)
    # DataLoader
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=TRAIN_BATCH_SIZE, num_workers=8, shuffle=True
    )
    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=VALID_BATCH_SIZE, num_workers=8
    )
    # Model
    model = FeedNetModel(
        n_features=xtrain.shape[1],
        n_targets=ytrain.shape[1],
        n_layers=params["n_layers"],
        hidden_size=params["hidden_size"],
        dropout=params["dropout"])
    model.to(device)
    # optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=params["learning_rate"])
#     SGD is slower than Adam
#     optimizer = torch.optim.SGD(model.parameters(
#     ), lr=params["learning_rate"], momentum=params["momentum"], nesterov=True)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=1, verbose=True)
    core = Core(model, optimizer, scheduler, device)
    best_loss = np.inf
    early_stopping_iter = 10
    early_stopping_counter = 0
    for epoch in range(EPOCHS):
        print(f'Fold {fold}')
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('-' * 10)
        train_loss = core.train_fn(train_data_loader)
        print(f'Train loss {train_loss}')
        val_loss = core.eval_fn(valid_data_loader)
        print(f'Val loss {val_loss}')
        print()
        if val_loss < best_loss:
            best_loss = val_loss
            if save_model:
                torch.save(model.state_dict(), f'model_{fold}.bin')
        else:
            early_stopping_counter += 1
        if early_stopping_counter > early_stopping_iter:
            break
    return best_loss