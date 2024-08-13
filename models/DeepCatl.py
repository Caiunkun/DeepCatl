import torch
import models.Transformer as TF
import torch.nn as nn
from fightingcv_attention.attention.ECAAttention import ECAAttention
class Attention(nn.Module):

    def __init__(self, channel=64, ratio=8):  #channel
        super(Attention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 1))

        self.shared_layer = nn.Sequential(
            nn.Linear(in_features=channel, out_features=channel // ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=channel // ratio, out_features=channel),
            nn.ReLU(inplace=True)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, F):
        b, c, _, _ = F.size()

        F_avg = self.shared_layer(self.avg_pool(F).reshape(b, c))
        F_max = self.shared_layer(self.max_pool(F).reshape(b, c))
        M = self.sigmoid(F_avg + F_max).reshape(b, c, 1, 1)

        return F * M

class deepcatl(nn.Module):

    def __init__(self):
        super(deepcatl, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.convolution_seq_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(4, 16), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=128)
        )
        self.convolution_shape_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(5, 16), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=128)
        )
        self.max_pooling_1 = nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 2))

        self.transformer_shape=TF.Transformer(101,8,8,128,128,0.1)

        self.attention_seq = Attention(channel=128, ratio=16)  # 32/4   64/8   128/16

        self.lstm = nn.LSTM(42,21,6, bidirectional=True, batch_first=True, dropout=0.2)
        self.convolution_seq_2 = nn.Sequential(
            nn.BatchNorm2d(num_features=128),
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 3), stride=(1, 1))
        )
        self.convolution_shape_2 = nn.Sequential(
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 3), stride=(1, 1))
        )
        self.output = nn.Sequential(
            nn.AdaptiveMaxPool2d(output_size=(1, 1)),
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=256, out_features=1),
            nn.Sigmoid()
        )

    def execute(self, seq, shape):

        shape = shape.float()
        shape=shape.squeeze(1)
        encoder_shape_output= self.transformer_shape(shape)
        encoder_shape_output = encoder_shape_output.unsqueeze(1)       #transformer
        conv_shape_1 = self.convolution_shape_1(encoder_shape_output)
        pool_shape_1 = self.max_pooling_1(conv_shape_1)
        pool_shape_1 = pool_shape_1.squeeze(2)
        out_shape, _ = self.lstm(pool_shape_1.to(self.device))        #lstm
        out_shape1 = out_shape.unsqueeze(2)
        conv_shape_2=self.convolution_shape_2(out_shape1)             #卷积
        #conv_shape_3=self.attention_seq(conv_shape_2)


        seq = seq.float()
        conv_seq_1 = self.convolution_seq_1(seq)
        pool_seq_1 = self.max_pooling_1(conv_seq_1)
        attention_seq_1 = self.attention_seq(pool_seq_1)
        attention_seq_2 = attention_seq_1.narrow(3, 0, 40)


        # conv_seq_1 = self.convolution_seq_1(seq.to(self.device)).to(self.device)
        # pool_seq_1= self.max_pooling_1(conv_seq_1)
        # conv_seq_2=self.convolution_seq_2(pool_seq_1)

        # torch.Size([64, 128, 1, 40])
        #conv_seq_3 = self.attention_seq(conv_seq_2)


        return self.output(torch.cat((conv_shape_2,attention_seq_2 ), dim=1))

    def forward(self, seq, shape):
        return self.execute(seq, shape)