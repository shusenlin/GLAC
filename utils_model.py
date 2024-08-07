import torch.nn as nn
import torch.nn.functional as F

class linear_model(nn.Module):
    def __init__(self, in_feature, out_feature ):
        super(linear_model, self).__init__()
        self.linear_layer = nn.Linear(in_feature,out_feature,bias=True)
        
        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
        self.linear_layer.apply(init_weights)
        
    def forward(self, input):
        out = self.linear_layer(input)
        return out

class mlp_model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(mlp_model, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = x.view(-1, self.num_flat_features(x))
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
