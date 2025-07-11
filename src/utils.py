import torch

class Trainer:
    def __init__(self, model, criterion, optimizer, device='mps' if torch.backends.mps.is_available() else 'cpu', without_mask = False):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.without_mask = without_mask
        self.device = device
        self.eval_device = torch.device('cpu') if not without_mask else self.device

        # Training statistics
        self.train_loss = 0
        self.val_loss = 0
        self.train_acc = 0
        self.val_acc = 0

    def init_stats_params(self):
        "Reset training statistics"
        self.train_loss = 0
        self.val_loss = 0
        self.train_acc = 0
        self.val_acc = 0

    def train_step(self, x, y, mask=None):
        "Training step"
        self.model.train()
        # MPS Computation
        self.model.to(self.device)
        x, y = x.to(self.device), y.to(self.device)

        if mask is not None:
            mask = mask.to(self.device)

        self.optimizer.zero_grad()

        if self.without_mask:
            y_pred = self.model(x)
        else:
            y_pred = self.model(x, mask)

        loss = self.criterion(y_pred, y)
        loss.backward()

        self.optimizer.step()

        # compute loss
        self.train_loss += loss.item()
        # compute accuracy
        pred = y_pred.argmax(dim=1, keepdim=True)
        correct = pred.eq(y.view_as(pred)).sum().item()
        self.train_acc += correct

    def train_step_bert(self, x, y, mask=None):
        "Training step"
        self.model.train()
        # MPS Computation
        self.model.to(self.device)
        x, y = x.to(self.device), y.to(self.device)

        if mask is not None:
            mask = mask.to(self.device)

        self.optimizer.zero_grad()

        if self.without_mask:
            y_pred = self.model(x)
        else:
            y_pred = self.model(x, mask)

        loss = self.criterion(y_pred.logits, y)
        loss.backward()

        self.optimizer.step()

        # compute loss
        self.train_loss += loss.item()
        # compute accuracy
        pred = y_pred.logits.argmax(dim=1, keepdim=True)
        correct = pred.eq(y.view_as(pred)).sum().item()
        self.train_acc += correct

    def val_step(self, x, y, mask=None):
        "Validation step"
        self.model.eval()
        # CPU Computation
        self.model.to(self.eval_device)
        x, y = x.to(self.eval_device), y.to(self.eval_device)
        if mask is not None:
            mask = mask.to(self.eval_device)

        if self.without_mask:
            y_pred = self.model(x)
        else:
            y_pred = self.model(x, mask)

        loss = self.criterion(y_pred, y)
        # compute loss
        self.val_loss += loss.item()
        # compute accuracy
        pred = y_pred.argmax(dim=1, keepdim=True)
        correct = pred.eq(y.view_as(pred)).sum().item()
        self.val_acc += correct

    def val_step_bert(self, x, y, mask=None):
        "Validation step"
        self.model.eval()
        # CPU Computation
        self.model.to(self.eval_device)
        x, y = x.to(self.eval_device), y.to(self.eval_device)
        if mask is not None:
            mask = mask.to(self.eval_device)

        if self.without_mask:
            y_pred = self.model(x)
        else:
            y_pred = self.model(x, mask)

        loss = self.criterion(y_pred.logits, y)
        # compute loss
        self.val_loss += loss.item()
        # compute accuracy
        pred = y_pred.logits.argmax(dim=1, keepdim=True)
        correct = pred.eq(y.view_as(pred)).sum().item()
        self.val_acc += correct

def test_models(text, model, tokenizer, device, bert_like = False, max_length=64):
    inputs = tokenizer(text, padding="max_length", truncation=True, max_length=max_length)
    src_mask = torch.tensor(inputs["attention_mask"], dtype=torch.bool, device=device).unsqueeze(0)
    src_inp = torch.tensor(inputs["input_ids"], dtype=torch.long, device=device).unsqueeze(0)
    with torch.inference_mode():
        outputs = model(src_inp, src_mask)
        if bert_like:
            outputs_prob = torch.nn.functional.softmax(outputs.logits, dim=1)
        else:
            outputs_prob = torch.nn.functional.softmax(outputs, dim=1)
        predictions = torch.argmax(outputs_prob, dim=1)
    return predictions
