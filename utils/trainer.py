from torch.utils.data import DataLoader
from utils.dataloader import DatasetClass, augment_until

import torch, joblib, tqdm

def fit(model, loader, optimizer, criterion, active, thresh):
    model.train()
    running_loss = .0
    failure_modes = {'x':[], 'y':[]}
    for batch in loader:
        optimizer.zero_grad()

        x = batch['x'].float().to(model.device)
        target = batch['y'].float().to(model.device)
        _, target = target.max(dim=1)

        pred = model(x)
        if active:
            errors = torch.abs(1 - pred[[u for u in range(target.shape[0])], target])
            failure_modes['x'].append(batch['x'][errors > thresh,:])
            failure_modes['y'].append(batch['y'][errors > thresh,:])

        loss = criterion(pred, target)
        loss = loss.mean()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    if active:
        for k, v in failure_modes.items():
            failure_modes[k] = torch.cat(v)
        
    return running_loss/len(loader.dataset), failure_modes

def validate(model, loader, criterion):
    model.eval()
    running_loss = .0
    for batch in loader:
        x = batch['x'].float().to(model.device)
        target = batch['y'].float().to(model.device)
        _, target = target.max(dim = 1)
        pred = model(x)

        loss = criterion(pred, target).mean()
        running_loss += loss.item()
    return running_loss/len(loader.dataset)


def train(model, data_train, data_test, optimizer, criterion, active = True, thresh = .7, n_epochs=200, log = 50, save_dir = 'weights/ckpt.pth'):
    train_loader= DataLoader(DatasetClass(data_train), batch_size=64)
    test_loader = DataLoader(DatasetClass(data_test), batch_size = 64)

    metrics = {'L_train' : [], 'L_val' : []}

    for i in tqdm.tqdm(range(n_epochs)):
        train_loss, failure_modes = fit(model, train_loader, optimizer, criterion, active, thresh)
        if active and i>50:
            augmented = augment_until(failure_modes, 300)
            acl_loader= DataLoader(DatasetClass(augmented), batch_size=64)
            _, _ = fit(model, acl_loader, optimizer, criterion, active, thresh)

        with torch.no_grad():
            val_loss = validate(model, test_loader, criterion)

        metrics['L_train'].append(train_loss)
        metrics['L_val'].append(val_loss)

        if i%log == 0:
            print(f'epoch : {i} - L_train : {train_loss} - L_val : {val_loss}')

        torch.save(model.state_dict(), save_dir)
    print('Done training.')
    return metrics


