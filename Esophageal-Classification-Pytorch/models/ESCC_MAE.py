# coding: UTF-8
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Config(object):

    """配置参数"""
    def __init__(self):
        self.model_name = 'ESCC_CNN'
        self.data_path = 'data/data.xlsx'                                # 训练集
        self.class_list = ["N", "T"]              # 类别名单                             # 词表
        self.save_path = 'saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.log_path = 'log/' + self.model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.dropout = 0.30                                             # 随机失活
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数            
        self.num_epochs = 50                                            # epoch数
        self.batch_size = 12                                            # mini-batch大小                                   
        self.learning_rate = 5e-4                                       # 学习率
        self.weight_decay = 0.005

        #Attetion
        self.seq_len = 712
        self.dim_model = 16
        self.hidden = 32
        self.num_head = 4



class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()

        self.pe = PositionalEncoding(config.dim_model, config.seq_len * 8)
        self.encoder_1 = Encoder(config.dim_model, config.num_head, config.hidden, config.dropout)
        self.encoder_2 = Encoder(config.dim_model, config.num_head, config.hidden, config.dropout)
        self.encoder_3 = Encoder(config.dim_model, config.num_head, config.hidden, config.dropout)
        self.encoder_4 = Encoder(config.dim_model, config.num_head, config.hidden, config.dropout)

        self.decoder_1 = Encoder(config.dim_model, config.num_head, config.hidden, config.dropout)
        self.pm_1 = PatchMerging(5696, config.dim_model)

        self.decoder_2 = Encoder(config.dim_model * 2, config.num_head, config.hidden * 2, config.dropout)
        self.pm_2 = PatchMerging(1424, config.dim_model * 2)

        self.decoder_3 = Encoder(config.dim_model * 4, config.num_head, config.hidden * 4, config.dropout)

        self.dropout_1 = nn.Dropout(config.dropout)
        self.fc_1 = nn.Linear(config.seq_len // 2, config.num_classes)
        self.fc_2 = nn.Linear(config.seq_len * 8, 5696)


    def forward(self, x):
        out = x[0]
        mask = x[1]
        mask = mask.unsqueeze(2)
        out = out.unsqueeze(2)

        out = out.repeat(1, 1, 16)     #[batch_size, 5696, 32]

        out = self.pe(out)
        out = torch.cat([self.encoder_1(out[:, i * 712:(i + 1) * 712, :], mask[:, i * 712:(i + 1) * 712, :]) for i in range(0, 8)], 1)
        
        out = torch.cat([out[:, i::4, :] for i in range(0, 4)], 1)
        mask = torch.cat([mask[:, i::4, :] for i in range(0, 4)], 1)

        out = self.pe(out)
        out = torch.cat([self.encoder_2(out[:, i * 712:(i + 1) * 712, :], mask[:, i * 712:(i + 1) * 712, :]) for i in range(0, 8)], 1)

        out = torch.cat([out[:, i::4, :] for i in range(0, 4)], 1)
        mask = torch.cat([mask[:, i::4, :] for i in range(0, 4)], 1)

        out = self.pe(out)
        out = torch.cat([self.encoder_3(out[:, i * 712:(i + 1) * 712, :], mask[:, i * 712:(i + 1) * 712, :]) for i in range(0, 8)], 1)
        
        out = torch.cat([out[:, i::4, :] for i in range(0, 4)], 1)
        mask = torch.cat([mask[:, i::4, :] for i in range(0, 4)], 1)

        out = self.pe(out)
        out = torch.cat([self.encoder_4(out[:, i * 712:(i + 1) * 712, :], mask[:, i * 712:(i + 1) * 712, :]) for i in range(0, 8)], 1)

        self.embed = out.detach()

        out = self.pe(out)
        out = torch.cat([self.decoder_1(out[:, i * 712:(i + 1) * 712, :]) for i in range(0, 8)], 1)
        out = self.pm_1(out)
        out = self.decoder_2(out)
        out = self.pm_2(out)
        out = self.decoder_3(out)

        out_1_1, _ = torch.max(out[:, :, 0:48], dim=2)
        out_1_2 = torch.mean(out[:, :, 0:48], dim=2)
        out_1 = self.fc_1((out_1_1 + out_1_2) / 2.0)

        out_2 = out[:, :, 48:64].flatten(start_dim=1)
        out_2 = self.fc_2(out_2)
        
        return out_1, out_2 
    
    def initialize(self):  # 初始化模型参数
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                if hasattr(m.bias, "data"):
                    m.bias.data.zero_()



class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, seq_len):
        #dim_model是每个词embedding后的维度
        super(PositionalEncoding, self).__init__()
        self.pe = torch.zeros(seq_len, dim_model)
        self.position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        self.div_term2 = torch.pow(torch.tensor(10000.0),torch.arange(0, dim_model, 2).float()/dim_model)
        self.div_term1 = torch.pow(torch.tensor(10000.0),torch.arange(1, dim_model, 2).float()/dim_model)
        #高级切片方式，即从0开始，两个步长取一个。即奇数和偶数位置赋值不一样。直观来看就是每一句话的
        self.pe[:, 0::2] = torch.sin(self.position * self.div_term2)
        self.pe[:, 1::2] = torch.cos(self.position * self.div_term1)
        #这里是为了与x的维度保持一致，释放了一个维度
        self.pe = self.pe.unsqueeze(0)
        self.register_buffer('positionalencoding', self.pe)
    def forward(self, x):
        pe = self.positionalencoding.repeat(x.size(0), 1, 1)
        x = x + pe
        return x



class Scaled_Dot_Product_Attention(nn.Module):
    '''Scaled Dot-Product Attention '''
    def __init__(self):
        super(Scaled_Dot_Product_Attention, self).__init__()

    def forward(self, Q, K, V, scale=None, mask=None):
        '''
        Args:
            Q: [batch_size, len_Q, dim_Q]
            K: [batch_size, len_K, dim_K]
            V: [batch_size, len_V, dim_V]
            scale: 缩放因子 论文为根号dim_K
        Return:
            self-attention后的张量，以及attention张量
        '''
        attention = torch.matmul(Q, K.permute(0, 2, 1))
        if scale:
            attention = attention * scale
        if mask != None:
            attention = attention.masked_fill_(mask == 0, -1e9)

        attention = F.softmax(attention, dim=-1)  
        context = torch.matmul(attention, V)
        return context



class Multi_Head_Attention(nn.Module):
    def __init__(self, dim_model, num_head, dropout=0.0):
        super(Multi_Head_Attention, self).__init__()
        self.num_head = num_head
        assert dim_model % num_head == 0
        self.dim_head = dim_model // self.num_head
        self.fc_Q = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_K = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_V = nn.Linear(dim_model, num_head * self.dim_head)
        self.attention = Scaled_Dot_Product_Attention()
        self.fc = nn.Linear(num_head * self.dim_head, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x, mask=None):
        batch_size = x.size(0)
        Q = self.fc_Q(x)
        K = self.fc_K(x)
        V = self.fc_V(x)
        Q = Q.view(batch_size * self.num_head, -1, self.dim_head)
        K = K.view(batch_size * self.num_head, -1, self.dim_head)
        V = V.view(batch_size * self.num_head, -1, self.dim_head)
        if mask != None:
            mask = mask.repeat(self.num_head, 1, 1)  
            mask = torch.bmm(mask, mask.permute(0, 2, 1))
        scale = K.size(-1) ** -0.5  # 缩放因子
        context = self.attention(Q, K, V, scale, mask)

        context = context.view(batch_size, -1, self.dim_head * self.num_head)
        out = self.fc(context)
        out = self.dropout(out)
        out = out + x  # 残差连接
        out = self.layer_norm(out)
        return out



class Position_wise_Feed_Forward(nn.Module):
    def __init__(self, dim_model, hidden, dropout=0.0):
        super(Position_wise_Feed_Forward, self).__init__()
        self.fc1 = nn.Linear(dim_model, hidden)
        self.fc2 = nn.Linear(hidden, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        out = self.fc1(x)
        out = F.elu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = out + x  # 残差连接
        out = self.layer_norm(out)
        return out



class Encoder(nn.Module):
    def __init__(self, dim_model, num_head, hidden, dropout):
        super(Encoder, self).__init__()
        self.attention = Multi_Head_Attention(dim_model, num_head, dropout)
        self.feed_forward = Position_wise_Feed_Forward(dim_model, hidden, dropout)

    def forward(self, x, mask=None):
        out = self.attention(x, mask)
        out = self.feed_forward(out)
        return out
    
    

class PatchMerging(nn.Module):
    def __init__(self, input_length, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_length = input_length
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=True)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        x0 = x[:, 0::4, :]  # B L/4 C
        x1 = x[:, 1::4, :]  # B L/4 C
        x2 = x[:, 2::4, :]  # B L/4 C
        x3 = x[:, 3::4, :]  # B L/4 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B L/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x