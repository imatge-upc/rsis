import torch
from torch import nn
import torch.nn.functional as f
from torch.autograd import Variable


class ConvLSTMCell(nn.Module):
    """
    Generate a convolutional LSTM cell
    """

    def __init__(self, args, input_size, hidden_size, kernel_size, padding):
        super(ConvLSTMCell,self).__init__()
        self.use_gpu = args.use_gpu
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, kernel_size, padding=padding)

    def forward(self, input_, prev_state, return_gates=False):

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

        if return_gates:
            return state, in_gate, remember_gate
        else:
            return state

class ConvGRUCell(nn.Module):
    """
    Generate a convolutional LSTM cell
    """

    def __init__(self, args, input_size, hidden_size, kernel_size, padding):
        super(ConvGRUCell,self).__init__()
        self.use_gpu = args.use_gpu
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Gates = nn.Conv2d(input_size + hidden_size, 2 * hidden_size, kernel_size, padding=padding)
        self.Update = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)

    def forward(self, input_, prev_state, return_gates=False):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            if self.use_gpu:
                prev_state = Variable(torch.zeros(state_size)).cuda()
            else:
                prev_state =Variable(torch.zeros(state_size))

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_state), 1)

        gates = self.Gates(stacked_inputs)
        # chunk across channel dimension
        z_gate, r_gate = gates.chunk(2, 1)

        # apply sigmoid non linearity
        z_gate = f.sigmoid(z_gate)
        r_gate = f.sigmoid(r_gate)

        stacked_inputs_update = torch.cat((input_,prev_state*r_gate),1)
        h = f.tanh(self.Update(stacked_inputs_update))

        state = (1-z_gate)*h + z_gate*prev_state

        if return_gates:
            return state, r_gate, z_gate
        else:
            return state

def _main():
    """
    Run some basic tests on the API
    """

    # define batch_size, channels, height, width
    b, c, h, w = 10, 3, 4, 8
    d = 5           # hidden state size
    lr = 1e-1       # learning rate
    T = 6           # sequence length
    max_epoch = 20  # number of epochs
    use_cuda = True
    # set manual seed
    torch.manual_seed(0)

    print('Instantiate model')
    model = ConvLSTMCell(c, d, kernel_size = 3, padding = 1)
    print(repr(model))
    model.cuda()

    print('Create input and target Variables')
    x = Variable(torch.rand(T, b, c, h, w)).cuda()
    y = Variable(torch.randn(T, b, d, h, w)).cuda()

    print('Create a MSE criterion')
    loss_fn = nn.MSELoss()

    if use_cuda:
        y.cuda()
        x.cuda()
        loss_fn.cuda()

    print('Run for', max_epoch, 'iterations')
    for epoch in range(0, max_epoch):
        state = None
        loss = 0
        for t in range(0, T):
            state = model(x[t], state)
            loss += loss_fn(state[0], y[t])

        print(' > Epoch {:2d} loss: {:.3f}'.format((epoch+1), loss.data[0]))

        # zero grad parameters
        model.zero_grad()

        # compute new grad parameters through time!
        loss.backward()

        # learning_rate step against the gradient
        for p in model.parameters():
            p.data.sub_(p.grad.data * lr)

    print('Input size:', list(x.data.size()))
    print('Target size:', list(y.data.size()))
    print('Last hidden state size:', list(state[0].size()))


if __name__ == '__main__':
    _main()


__author__ = "Alfredo Canziani"
__credits__ = ["Alfredo Canziani"]
__maintainer__ = "Alfredo Canziani"
__email__ = "alfredo.canziani@gmail.com"
__status__ = "Prototype"  # "Prototype", "Development", or "Production"
__date__ = "Jan 17"
