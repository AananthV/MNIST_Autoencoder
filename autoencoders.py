import torch.nn as nn

class AE_3D_100(nn.Module):
    def __init__(self, n_features=4):
        super(AE_3D_100, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 100),
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100, 50),
            nn.Tanh(),
            nn.Linear(50, 3)
        )
        
        self.decoder = nn.Sequential(
            nn.Tanh(),
            nn.Linear(3, 50),
            nn.Tanh(),
            nn.Linear(50, 100),
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100, n_features)
        )
        
    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    def describe(self):
        return 'in-100-100-50-3-50-100-100-out'

class AE_big(nn.Module):
    def __init__(self, n_features=4):
        super(AE_big, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 8),
            nn.Tanh(),
            nn.Linear(8, 6),
            nn.Tanh(),
            nn.Linear(6, 4),
            nn.Tanh(),
            nn.Linear(4, 3)
        )
        self.decoder = nn.Sequential(
            nn.Tanh(),
            nn.Linear(3, 4),
            nn.Tanh(),
            nn.Linear(4, 6),
            nn.Tanh(),
            nn.Linear(6, 8),
            nn.Tanh(),
            nn.Linear(8, n_features)
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    def describe(self):
        return 'in-8-6-4-3-4-6-8-out'