class Core:
    def __init__(self, model, optimizer, scheduler, device):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
    @staticmethod
    def loss_fn(outputs, targets):
        return nn.BCEWithLogitsLoss()(outputs, targets)
    def train_fn(self, data_loader):
        self.model.train()
        lossess = 0
        for d in data_loader:
            self.optimizer.zero_grad()
            inputs = d["x"].to(self.device)
            targets = d["y"].to(self.device)
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            self.optimizer.step()
            lossess += loss.item()
        self.scheduler.step(loss)
        return lossess/len(data_loader)
    def eval_fn(self, data_loader):
        self.model.eval()
        lossess = 0
        for d in data_loader:
            inputs = d["x"].to(self.device)
            targets = d["y"].to(self.device)
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            lossess += loss.item()
        return lossess/len(data_loader)