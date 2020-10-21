class RKDataset:
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets
    def __len__(self):
        return self.inputs.shape[0]
    def __getitem__(self, item):
        return {
            'x': torch.tensor(self.inputs[item, :], dtype=torch.float),
            'y': torch.tensor(self.targets[item, :], dtype=torch.float)
        }