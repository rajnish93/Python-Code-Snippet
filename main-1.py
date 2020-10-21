def main_fn(fold, params, save_model=False):
    feature_columns = "description"
    targets_columns = "label"
    train_df = df[df.kfold != fold].reset_index(drop=True)
    valid_df = df[df.kfold == fold].reset_index(drop=True)
    xtrain = train_df[feature_columns].to_numpy()
    ytrain = train_df[targets_columns].to_numpy()
    xvalid = valid_df[feature_columns].to_numpy()
    yvalid = valid_df[targets_columns].to_numpy()
    # Dataset
    train_dataset = RKDataset(reviews=xtrain, targets=ytrain)
    valid_dataset = RKDataset(reviews=xvalid, targets=yvalid)
    # DataLoader
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=TRAIN_BATCH_SIZE, num_workers=8, shuffle=True
    )
    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=VALID_BATCH_SIZE, num_workers=8
    )
    # length of dataloder’s dataset
    # print(f"train_loader_size{len(train_data_loader.dataset)}")
#     print(f"valid_loader_size{len(valid_data_loader)}")
#     print(f"train_size{len(train_df)}")
#     print(f"xtrain_size{len(xtrain)}")
    # Model
    model = RKClassifier(n_classes=3, dropout=params["dropout"])
    model.to(device)
    # optimizer
#     optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
    optimizer = torch.optim.SGD(model.parameters(
    ), lr=params["learning_rate"], momentum=params["momentum"], nesterov=True)
#     LEARNING SCHEDULER
    scheduler = ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=0, verbose=True)
    core = Core(model, optimizer, scheduler, device)
    best_loss = np.inf
    early_stopping_iter = 10
    early_stopping_counter = 0
    # history = defaultdict(list)
    # best_accuracy = 0
    for epoch in range(EPOCHS):
        print(f'Fold {fold}')
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('-' * 10)
        train_acc, train_loss = core.train_fn(train_data_loader)
        print(f'Train loss {train_loss} accuracy {train_acc}')
        val_acc, val_loss = core.eval_fn(valid_data_loader)
        print(f'Val   loss {val_loss} accuracy {val_acc}')
        print()
        # history['train_acc'].append(train_acc)
        # history['train_loss'].append(train_loss)
        # history['val_acc'].append(val_acc)
        # history['val_loss'].append(val_loss)
        # if val_acc > best_accuracy:
        #     torch.save(model.state_dict(), 'best_model_state.bin')
        #     best_accuracy = val_acc
        if val_loss < best_loss:
            best_loss = val_loss
            if save_model:
                torch.save(model.state_dict(), f'model_{fold}.bin')
        else:
            early_stopping_counter += 1
        if early_stopping_counter > early_stopping_iter:
            break
    return best_loss