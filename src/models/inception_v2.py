import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv1d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm1d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class BasicFC(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(BasicFC, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.fc(x)
        return F.relu(x, inplace=True)


class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()

        self.branch1 = BasicConv1d(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            BasicConv1d(in_channels, ch3x3red, kernel_size=1),
            BasicConv1d(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            BasicConv1d(in_channels, ch5x5red, kernel_size=1),
            BasicConv1d(ch5x5red, ch5x5, kernel_size=7, padding=3)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            BasicConv1d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


class SingleInceptionModel_v2(nn.Module):
    def __init__(self, obs_dim, fnl_dim, num_dim, lag, dropout, added_point, reg):
        super(SingleInceptionModel_v2, self).__init__()
        obs_time = lag * 4
        fnl_time = lag * 4 - 2

        self.obs_inception = Inception(obs_dim, 64, 96, 128, 16, 32, 32)
        self.obs_inception_dropout = nn.Dropout(dropout)

        self.fnl_inception = Inception(fnl_dim, 64, 96, 128, 16, 32, 32)
        self.fnl_inception_dropout = nn.Dropout(dropout)

        self.num_rnn = getattr(nn, "GRU")(num_dim, num_dim, 1, True, batch_first=True)

        self.fnl_resize = BasicConv1d(fnl_time, obs_time, kernel_size=1)
        self.fnl_resize_dropout = nn.Dropout(dropout)

        if added_point:
            num_fc_dim = num_dim * 2
        else:
            num_fc_dim = num_dim

        # print(lag, num_fc_dim, 256 * (lag * 4) + num_fc_dim)

        self.fc1 = BasicFC(256 * 2 * (lag * 4) + num_fc_dim, 256)
        self.fc1_dropout = nn.Dropout(dropout)

        if reg:
            self.fc2 = nn.Linear(256, 1)
        else:
            self.fc2 = nn.Linear(256, 4)

    def forward(self, x_obs, x_fnl, x_num=None, point_num=None):
        # first conv (fnl, obs seperately)
        x_obs = self.obs_inception(x_obs)
        x_fnl = self.fnl_inception(x_fnl)

        # resize fnl
        x_fnl = x_fnl.transpose(1, 2)
        x_fnl = self.fnl_resize(x_fnl)
        x_fnl = x_fnl.transpose(1, 2)

        # concat fnl,obs and flattening
        x = torch.cat((x_obs, x_fnl), dim=1)
        x = x.view(x.shape[0], -1)

        # print(x.shape, x_num.shape, point_num.shape)

        # flattening numeric data and concat
        if x_num is not None:
            _, out = self.num_rnn(x_num)
            out = out.squeeze()
            # print(out.shape)
            if point_num is not None:
                out = torch.cat((out, point_num), dim=1)
            x = torch.cat((x, out), dim=1)
        # print(x.shape)
        # fully connected parts
        x = self.fc1(x)
        x = self.fc1_dropout(x)
        x = self.fc2(x)
        return x


class DoubleInceptionModel_v2(nn.Module):
    def __init__(self, obs_dim, fnl_dim, num_dim, lag, dropout, added_point, reg):
        super(DoubleInceptionModel_v2, self).__init__()
        obs_time = lag * 4
        fnl_time = lag * 4 - 2

        self.obs_inception = Inception(obs_dim, 64, 96, 128, 16, 32, 32)
        self.obs_inception_dropout = nn.Dropout(dropout)

        self.fnl_inception = Inception(fnl_dim, 64, 96, 128, 16, 32, 32)
        self.fnl_inception_dropout = nn.Dropout(dropout)

        self.num_rnn = getattr(nn, "GRU")(num_dim, num_dim, 1, True, batch_first=True)

        self.fnl_resize = BasicConv1d(fnl_time, obs_time, kernel_size=1)
        self.fnl_resize_dropout = nn.Dropout(dropout)

        self.second_inception = Inception(512, 128, 192, 256, 32, 64, 64)
        self.second_inception_dropout = nn.Dropout(dropout)

        if added_point:
            num_fc_dim = num_dim * 2
        else:
            num_fc_dim = num_dim

        self.fc1 = BasicFC(256 * 2 * (lag * 4) + num_fc_dim, 256)
        self.fc1_dropout = nn.Dropout(dropout)

        if reg:
            self.fc2 = nn.Linear(256, 1)
        else:
            self.fc2 = nn.Linear(256, 4)

    def forward(self, x_obs, x_fnl, x_num=None, point_num=None):
        # first conv (fnl, obs seperately)
        x_obs = self.obs_inception(x_obs)
        x_fnl = self.fnl_inception(x_fnl)

        # resize fnl
        x_fnl = x_fnl.transpose(1, 2)
        x_fnl = self.fnl_resize(x_fnl)
        x_fnl = x_fnl.transpose(1, 2)

        # concat fnl,obs
        x = torch.cat((x_obs, x_fnl), dim=1)
        # print(x.shape)
        # second conv
        x = self.second_inception(x)
        # print(x.shape)
        # flattening
        x = x.view(x.shape[0], -1)

        # flattening numeric data and concat
        if x_num is not None:
            _, out = self.num_rnn(x_num)
            out = out.squeeze()
            if point_num is not None:
                out = torch.cat((out, point_num), dim=1)
            x = torch.cat((x, out), dim=1)
        # print(x.shape)
        # fully connected parts
        x = self.fc1(x)
        x = self.fc1_dropout(x)
        x = self.fc2(x)
        return x
