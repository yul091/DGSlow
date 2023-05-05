import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


class EncoderRNN(nn.Module):
    def __init__(
        self, 
        input_size: int = 10, 
        hidden_size: int = 256,
        device: Optional[torch.device] = None,
    ):
        super(EncoderRNN, self).__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.linear = nn.Linear(hidden_size, input_size)
        
    def forward(self, input, hidden):
        embedded = self.embedding(input)
        output, hidden = self.gru(embedded, hidden) # B X T X H
        output = self.linear(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)
    
    
    
    
class AttnDecoderRNN(nn.Module):
    def __init__(
        self, 
        hidden_size: int = 256, 
        output_size: int = 10, 
        dropout_p: float = 0.1, 
        max_length: int = 1024,
        device: Optional[torch.device] = None,
    ):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)
    


def pareto_step(weights_list, model_gradients):
    M1 = np.matmul(model_gradients, np.transpose(model_gradients))
    e = np.mat(np.ones(np.shape(weights_list)))
    M = np.hstack((M1,np.transpose(e)))
    mid = np.hstack((e, np.mat(np.zeros((1,1)))))
    M = np.vstack((M,mid))
    z = np.mat(np.zeros(np.shape(weights_list)))
    nid = np.hstack((z, np.mat(np.ones((1,1)))))
    w = np.matmul(np.matmul(M,np.linalg.inv(np.matmul(M, np.transpose(M)))), np.transpose(nid))
    if len(w) > 1:
        w = np.transpose(w)
        w = w[0,0:np.shape(w)[1]]
        mid = np.where(w > 0, 1.0, 0)
        nid = np.multiply(mid, w)
        uid = sorted(nid[0].tolist()[0], reverse=True)
        sv = np.cumsum(uid)
        rho = np.where(uid > (sv - 1.0) / range(1, len(uid)+1), 1.0, 0.0)
        r = max(np.argwhere(rho))
        theta = max(0, (sv[r] - 1.0) / (r+1))
        w = np.where(nid - theta>0.0, nid - theta, 0)
    return w
    
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    V = 2567
    H = 256
    encoder1 = EncoderRNN(V, H).to(device)
    encoder2 = EncoderRNN(V, H).to(device)
    # decoder = AttnDecoderRNN(H, V).to(device)
    criterion = nn.NLLLoss()
    
    B, T = 3, 20
    inputs = torch.randint(0, V, (B, T)).to(device)
    hidden = torch.randn(1, T, H).to(device)
    labels = torch.randint(0, V, (B, T)).to(device)
    output1, hidden1 = encoder1(inputs, hidden) # B X T X V
    loss1 = criterion(output1.view(-1, V), labels.view(-1))
    loss1.backward()
    grad1 = encoder1.embedding.weight.grad # V X H
    print("grad1", grad1.shape)
    
    output2, hidden2 = encoder2(inputs, hidden) # B X T X V
    loss2 = criterion(output2.view(-1, V), labels.view(-1))
    loss2.backward()
    grad2 = encoder2.embedding.weight.grad # V X H
    print("grad2", grad2.shape)
    
    # Combien gradients
    w1, w2 = 0.5, 0.5
    weights = np.mat([w1, w2])
    # gradient_sum = torch.cat((grad1.view(-1), grad2.view(-1)), 0) # V * H * 2
    grad_sum1 = torch.randn(2*V*H)
    grad_sum2 = torch.randn(2*V*H)
    print("gradient_sum", grad_sum1.shape)
    grad_paras_tensor = torch.stack((grad_sum1, grad_sum2), 0) # 2 X (V * H * 2)
    grad_paras = grad_paras_tensor.cpu().numpy()
    print("grad_paras", grad_paras.shape)
    mid = pareto_step(weights, grad_paras)
    new_w1, new_w2 = mid[0,0], mid[0,1]
    print("new w1 {}, new w2 {}".format(new_w1, new_w2))
    
    
    