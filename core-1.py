class Core:
    #  not using scheduler in every batch instead use after every eoch
    def __init__(self, model, optimizer, scheduler, device):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
    @staticmethod
    def loss_fn(outputs, targets):
        return nn.CrossEntropyLoss()(outputs, targets)
    def train_fn(self, data_loader):
        self.model.train()
        losses = []
        correct_predictions = 0
        correct = 0
        for d in data_loader:
            self.optimizer.zero_grad()
            input_ids = d["input_ids"].to(self.device)
            attention_mask = d["attention_mask"].to(self.device)
            targets = d["targets"].to(self.device)
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)
            loss = self.loss_fn(outputs, targets)
#             correct_predictions += torch.sum(preds == targets)
            correct += (preds == targets).sum().item()
            losses.append(loss.item())
            loss.backward()
            self.optimizer.step()
        accuracy = 100 * correct / len(data_loader.dataset)
        self.scheduler.step(accuracy)
        return accuracy, np.mean(losses)
#         return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)
    def eval_fn(self, data_loader):
        self.model.eval()
        losses = []
        correct_predictions = 0
        for d in data_loader:
            input_ids = d["input_ids"].to(self.device)
            attention_mask = d["attention_mask"].to(self.device)
            targets = d["targets"].to(self.device)
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)
            loss = self.loss_fn(outputs, targets)
            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())
        return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)