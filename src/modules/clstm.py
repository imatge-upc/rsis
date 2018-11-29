import torch
from torch import nn
import torch.nn.functional as f
from torch.autograd import Variable
from modules.coordconv import CoordConv


class ConvLSTMCell(nn.Module):
    """
    Generate a convolutional LSTM cell
    """

    def __init__(self, args, input_size, hidden_size, kernel_size, padding):
        super(ConvLSTMCell, self).__init__()
        self.use_gpu = args.use_gpu
        self.input_size = input_size
        self.hidden_size = hidden_size
        if args.coordconv:
            self.Gates = CoordConv(in_channels=input_size + hidden_size + 2,
                                   out_channels=4 * hidden_size,
                                   kernel_size=kernel_size, padding=padding)
        else:

            self.Gates = nn.Conv2d(in_channels=input_size + hidden_size,
                                   out_channels=4 * hidden_size,
                                   kernel_size=kernel_size, padding=padding)
        #self.Gates.bias.data.fill_(1.0)

    def forward(self, input_, prev_state):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            if self.use_gpu:
                prev_state = (
                    Variable(torch.zeros(state_size)).cuda(),
                    Variable(torch.zeros(state_size)).cuda()
                )
            else:
                prev_state = (
                    Variable(torch.zeros(state_size)),
                    Variable(torch.zeros(state_size))
                )

        prev_hidden, prev_cell = prev_state

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = f.sigmoid(in_gate)
        remember_gate = f.sigmoid(remember_gate)
        out_gate = f.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = f.tanh(cell_gate)
        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * f.tanh(cell)

        state = [hidden,cell]

        return state
