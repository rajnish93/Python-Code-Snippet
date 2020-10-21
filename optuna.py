# optuna objective function
def objective(trial):
    params = {
        "dropout": trial.suggest_uniform("dropout", 0.1, 0.7),
        "n_layers": trial.suggest_int("n_layers", 1, 7),
        "hidden_size": trial.suggest_int("hidden_size", 15, 2048),
        #         "momentum": trial.suggest_uniform("momentum", 0.1, 0.9),
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-6, 1e-3)
    }
    all_losses = []
    for f in range(5):
        temp_loss = main_fn(f, params, save_model=False)
        all_losses.append(temp_loss)
    return np.mean(all_losses)
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)
print("best trial")
trial_ = study.best_trial
# print(trial_.values)
print(trial_.params)
# save model
scores = 0
for j in range(5):
    src = main_fn(j, trial_.params, save_model=True)
    scores += src
print(scores/5)