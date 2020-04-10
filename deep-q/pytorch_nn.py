import torch.nn as nn
import torch as T
import torch.nn.functional as F
import torch.optim as optim

class LinearClassifier(nn.Module):
    def __init__(self, lr, input_dim, classes):
        super(LeanerClassifer, self)
        self.lr = lr
        self.fc1 = nn.Linear(*input_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, classes)

        self.optim = optim.Adam(self.parameters, lr= lr)
        self.loss = nn.CrossEntropyLoss()
        self.device = T.device('cuda:0' if T.cuda.is_available else 'cpu')
        self.to(self.device)


    def forward (self, data):
        layer1 = F.sigmoid(self.fc1(data))
        layer2 = F.sigmoid(self.fc2(layer1))
        layer3 = self.fc3(layer2)

        return layer3

    def learn(self, data, labels):
        self.optim.zero_grad()
        data = T.tensor(data).to(self.device)
        labels = T.tensor(labels).to(self.device)
        predictions = self.forward(data)

        cost = loss(predictions, labels)

        cost.backward()

        self.optim.step()
        
