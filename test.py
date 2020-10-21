# test accuracy of model
test_acc, _ = eval_fn(
    model,
    test_data_loader,
    loss_fn,
    device,
    len(df_test)
)
test_acc.item()