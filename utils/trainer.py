from torch.utils.data import DataLoader
from utils.dataloader import DatasetClass

import torch, joblib, tqdm

def fit(model, loader, optimizer, criterion):
    model.train()
    running_loss = .0
    for batch in loader:
        optimizer.zero_grad()

        x = batch['x'].float().to(model.device)
        target = batch['y'].float().to(model.device)
        _, target = target.max(dim=1)

        pred = model(x)

        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss/len(loader.dataset)

def validate(model, loader, criterion):
    model.eval()
    running_loss = .0
    for batch in loader:
        x = batch['x'].float().to(model.device)
        target = batch['y'].float().to(model.device)
        _, target = target.max(dim = 1)
        pred = model(x)

        loss = criterion(pred, target)
        running_loss += loss.item()
    return running_loss/len(loader.dataset)


def train(model, data_train, data_test, optimizer, criterion, n_epochs, log = 50, save_dir = 'weights/ckpt.pth'):
    train_loader= DataLoader(DatasetClass(data_train), batch_size=64)
    test_loader = DataLoader(DatasetClass(data_test), batch_size = 64)

    metrics = {'L_train' : [], 'L_val' : []}

    for i in tqdm.tqdm(range(n_epochs)):
        train_loss = fit(model, train_loader, optimizer, criterion)
        with torch.no_grad():
            val_loss = validate(model, test_loader, criterion)

        metrics['L_train'].append(train_loss)
        metrics['L_val'].append(val_loss)

        if i%log == 0:
            print(f'epoch : {i} - L_train : {train_loss} - L_val : {val_loss}')

        torch.save(model.state_dict(), save_dir)
    print('Done training.')
    return metrics


