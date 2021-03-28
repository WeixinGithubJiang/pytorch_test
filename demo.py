import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from PIL import Image




class dataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        return np.array([x]), np.array([y])
    
    
class Net(torch.nn.Module):
    def __init__(self, order=3):
        super(Net, self).__init__()
        self.order = order
        self.params = torch.nn.Parameter(torch.randn(self.order, dtype=torch.float32), requires_grad=True)
        
        
    def forward(self, x):
        out = 0
        for i in range(self.order):
            out = out + self.params[i]*x**i
        return out
    
if __name__ == "__main__":
    x = np.linspace(-1,1,10000000)
    order = 10
    params = np.random.normal(size=order)

    y = 0
    for i in range(order):
        y = y + params[i]*x**i
    y = y + np.random.normal(scale=0.02,size=len(x))
    
    model = Net(order*2)
    device = torch.device("cuda:0")
    model = torch.nn.DataParallel(model, device_ids=[0,1])
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)
    
    test_dataset = dataset(x,y)
    train_dataloader = torch.utils.data.DataLoader(test_dataset, 
                                                  batch_size=1000000, 
                                                 shuffle=True,
                                                 num_workers=12)
    loss_history = []
    model.train()
    for epoch in range(10):
        for idx, (inpt, target) in enumerate(train_dataloader):
            optimizer.zero_grad()
            pred = model(inpt.to(device))
            loss = torch.nn.MSELoss()(pred, target.to(device))
            loss.backward()
            optimizer.step()
            print(epoch, idx, loss.item())
            loss_history.append(loss.item())
            
    plt.plot(loss_history)
    plt.savefig("loss_history.png")
    plt.close()
    
    test_dataloader = torch.utils.data.DataLoader(test_dataset, 
                                              batch_size=10000000, 
                                             shuffle=False,
                                             num_workers=12)
    inpt, target = next(iter(test_dataloader))
    model.eval()
    with torch.no_grad():
        pred = model(inpt.to(device))
        
    plt.plot(x, y, label="gt")
    plt.plot(x, pred.data.cpu().numpy().reshape(-1), label="pred")
    plt.legend()
    plt.savefig("pred_vs_gt.png")
    plt.close()

