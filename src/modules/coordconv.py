import torch
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AddCoords(nn.Module):
    def __init__(self, radius_channel=False):
        super(AddCoords, self).__init__()
        self.radius_channel = radius_channel

    def forward(self, in_tensor):
        """
        in_tensor: (batch_size, channels, x_dim, y_dim)
        [0,0,0,0]   [0,1,2,3]
        [1,1,1,1]   [0,1,2,3]    << (i,j)th coordinates of pixels added as separate channels
        [2,2,2,2]   [0,1,2,3]
        taken from mkocabas.
        """
        batch_size_tensor = in_tensor.shape[0]

        xx_ones = torch.ones([1, in_tensor.shape[2]], dtype=torch.int32)
        xx_ones = xx_ones.unsqueeze(-1)

        xx_range = torch.arange(in_tensor.shape[2], dtype=torch.int32).unsqueeze(0)
        xx_range = xx_range.unsqueeze(1)

        xx_channel = torch.matmul(xx_ones, xx_range)
        xx_channel = xx_channel.unsqueeze(-1)

        yy_ones = torch.ones([1, in_tensor.shape[3]], dtype=torch.int32)
        yy_ones = yy_ones.unsqueeze(1)

        yy_range = torch.arange(in_tensor.shape[3], dtype=torch.int32).unsqueeze(0)
        yy_range = yy_range.unsqueeze(-1)

        yy_channel = torch.matmul(yy_range, yy_ones)
        yy_channel = yy_channel.unsqueeze(-1)

        xx_channel = xx_channel.permute(0, 3, 1, 2)
        yy_channel = yy_channel.permute(0, 3, 1, 2)

        xx_channel = xx_channel.float() / (in_tensor.shape[2] - 1)
        yy_channel = yy_channel.float() / (in_tensor.shape[3] - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size_tensor, 1, 1, 1)
        yy_channel = yy_channel.repeat(batch_size_tensor, 1, 1, 1)

        out = torch.cat([in_tensor.to(device), xx_channel.to(device), yy_channel.to(device)], dim=1)

        if self.radius_channel:
            radius_calc = torch.sqrt(torch.pow(xx_channel - 0.5, 2) + torch.pow(yy_channel - 0.5, 2))
            out = torch.cat([out, radius_calc], dim=1).to(device)

        return out


class CoordConv(nn.Module):
    """ add any additional coordinate channels to the input tensor """
    def __init__(self, radius_channel=False, *args, **kwargs):
        super(CoordConv, self).__init__()
        self.addcoord = AddCoords(radius_channel=radius_channel)
        self.conv = nn.Conv2d(*args, **kwargs)

    def forward(self, in_tensor):
        out = self.addcoord(in_tensor)
        out = self.conv(out)
        return out


class CoordConvTranspose(nn.Module):
    """CoordConvTranspose layer for segmentation tasks."""
    def __init__(self, radius_channel, *args, **kwargs):
        super(CoordConv, self).__init__()
        self.addcoord = AddCoords(radius_channel=radius_channel)
        self.convT = nn.ConvTranspose2d(*args, **kwargs)

    def forward(self, in_tensor):
        out = self.addcoord(in_tensor)
        out = self.convT(out)
        return out