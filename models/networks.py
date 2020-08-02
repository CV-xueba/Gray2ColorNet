import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, bias_input_nc, output_nc, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)
    net = Gray2ColorNet(input_nc, bias_input_nc, output_nc, norm_layer=norm_layer)

    return init_net(net, init_type, init_gain, gpu_ids)


class ResBlock(nn.Module):
    def __init__(self, dim, norm_layer, use_dropout, use_bias):
        super(ResBlock, self).__init__()
        conv_block = [nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=use_bias), norm_layer(dim)]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)  # add skip connections
        return out


class global_network(nn.Module):
    def __init__(self, in_dim):
        super(global_network, self).__init__()
        model = [nn.Conv2d(in_dim, 512, kernel_size=1, padding=0), nn.ReLU(True)]
        model += [nn.Conv2d(512, 512, kernel_size=1, padding=0), nn.ReLU(True)]
        model += [nn.Conv2d(512, 512, kernel_size=1, padding=0), nn.ReLU(True)]
        self.model = nn.Sequential(*model)

        self.model_1 = nn.Sequential(*[nn.Conv2d(512, 512, kernel_size=1, padding=0), nn.ReLU(True)])
        self.model_2 = nn.Sequential(*[nn.Conv2d(512, 256, kernel_size=1, padding=0), nn.ReLU(True)])
        self.model_3 = nn.Sequential(*[nn.Conv2d(512, 128, kernel_size=1, padding=0), nn.ReLU(True)])

    def forward(self, x):
        x = self.model(x)
        x1 = self.model_1(x)
        x2 = self.model_2(x)
        x3 = self.model_3(x)

        return x1, x2, x3


class ref_network(nn.Module):
    def __init__(self, norm_layer):
        super(ref_network, self).__init__()
        model1 = [nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1, bias=True), nn.ReLU(True)]
        model1 += [nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True), nn.ReLU(True), norm_layer(64)]
        self.model1 = nn.Sequential(*model1)
        model2 = [nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True), nn.ReLU(True)]
        model2 += [nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True), nn.ReLU(True), norm_layer(128)]
        self.model2 = nn.Sequential(*model2)
        model3 = [nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True), nn.ReLU(True)]
        model3 += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True), nn.ReLU(True), norm_layer(128)]
        self.model3 = nn.Sequential(*model3)
        model4 = [nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True), nn.ReLU(True)]
        model4 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True), nn.ReLU(True), norm_layer(256)]
        self.model4 = nn.Sequential(*model4)

    def forward(self, color, corr, H, W):
        conv1 = self.model1(color)
        conv2 = self.model2(conv1[:,:,::2,::2])
        conv2_flatten = conv2.view(conv2.shape[0], conv2.shape[1], -1)

        align = torch.bmm(conv2_flatten, corr)
        align_1 = align.view(align.shape[0], align.shape[1], H, W)
        align_2 = self.model3(align_1[:,:,::2,::2])
        align_3 = self.model4(align_2[:,:,::2,::2])

        return align_1, align_2, align_3


class conf_feature(nn.Module):
    def __init__(self):
        super(conf_feature, self).__init__()
        self.fc1 = nn.Sequential(*[nn.Conv1d(4096, 1024, kernel_size=1, stride=1, padding=0, bias=True), nn.ReLU(True)])
        self.fc2 = nn.Sequential(*[nn.Conv1d(1024, 1, kernel_size=1, stride=1, padding=0, bias=True), nn.Sigmoid()])

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class classify_network(nn.Module):
    def __init__(self):
        super(classify_network, self).__init__()
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Linear(512, 1000)

    def forward(self, x):
        x = self.maxpool(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.fc(x)
        return x


class Gray2ColorNet(nn.Module):
    def __init__(self, input_nc, bias_input_nc, output_nc, norm_layer=nn.BatchNorm2d):
        super(Gray2ColorNet, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        use_bias = True

        downmodel1=[nn.Conv2d(input_nc, 64, kernel_size=3, stride=1, padding=1, bias=use_bias), nn.ReLU(True)]
        downmodel1+=[nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias), nn.ReLU(True), norm_layer(64)]

        downmodel2=[nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=use_bias), nn.ReLU(True)]
        downmodel2+=[nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=use_bias), nn.ReLU(True), norm_layer(128)]

        downmodel3=[nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=use_bias), nn.ReLU(True)]
        downmodel3+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias), nn.ReLU(True)]
        downmodel3+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias), nn.ReLU(True), norm_layer(256)]

        downmodel4=[nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=use_bias), nn.ReLU(True)]
        downmodel4+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias), nn.ReLU(True)]
        downmodel4+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias), nn.ReLU(True), norm_layer(512)]

        downmodel5=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=use_bias), nn.ReLU(True)]
        downmodel5+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=use_bias), nn.ReLU(True)]
        downmodel5+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=use_bias), nn.ReLU(True), norm_layer(512)]

        downmodel6=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=use_bias), nn.ReLU(True)]
        downmodel6+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=use_bias), nn.ReLU(True)]
        downmodel6+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=use_bias), nn.ReLU(True), norm_layer(512)]

        resblock0_1 = [nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=use_bias), norm_layer(512), nn.ReLU(True)]
        self.resblock0_2 = ResBlock(512, norm_layer, False, use_bias)
        self.resblock0_3 = ResBlock(512, norm_layer, False, use_bias)

        upmodel1up=[nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=use_bias)]
        upmodel1short=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        upmodel1=[nn.ReLU(True), nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias), nn.ReLU(True)]
        upmodel1+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias), nn.ReLU(True), norm_layer(256)]

        resblock1_1 = [nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=use_bias), norm_layer(256), nn.ReLU(True)]
        self.resblock1_2 = ResBlock(256, norm_layer, False, use_bias)
        self.resblock1_3 = ResBlock(256, norm_layer, False, use_bias)

        upmodel2up=[nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=use_bias),]
        upmodel2short=[nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        upmodel2=[nn.ReLU(True), nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=use_bias), nn.ReLU(True), norm_layer(128)]

        resblock2_1 = [nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=use_bias), norm_layer(128), nn.ReLU(True)]
        self.resblock2_2 = ResBlock(128, norm_layer, False, use_bias)
        self.resblock2_3 = ResBlock(128, norm_layer, False, use_bias)
        
        upmodel3up=[nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, bias=use_bias),]
        upmodel3short=[nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        upmodel3=[nn.ReLU(True), nn.Conv2d(128, 128, kernel_size=3, dilation=1, stride=1, padding=1, bias=use_bias), nn.LeakyReLU(negative_slope=.2)]

        self.global_network = global_network(bias_input_nc)
        self.ref_network = ref_network(norm_layer)
        self.conf_feature = conf_feature()
        self.classify_network = classify_network()

        model_out1 = [nn.Conv2d(256, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=use_bias), nn.Tanh()]
        model_out2 = [nn.Conv2d(128, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=use_bias), nn.Tanh()]
        model_out3 = [nn.Conv2d(128, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=use_bias), nn.Tanh()]

        self.model1 = nn.Sequential(*downmodel1)
        self.model2 = nn.Sequential(*downmodel2)
        self.model3 = nn.Sequential(*downmodel3)
        self.model4 = nn.Sequential(*downmodel4)
        self.model5 = nn.Sequential(*downmodel5)
        self.model6 = nn.Sequential(*downmodel6)
        self.model8up = nn.Sequential(*upmodel1up)
        self.model8 = nn.Sequential(*upmodel1)
        self.model9up = nn.Sequential(*upmodel2up)
        self.model9 = nn.Sequential(*upmodel2)
        self.model10up = nn.Sequential(*upmodel3up)
        self.model10 = nn.Sequential(*upmodel3)
        self.model3short8 = nn.Sequential(*upmodel1short)
        self.model2short9 = nn.Sequential(*upmodel2short)
        self.model1short10 = nn.Sequential(*upmodel3short)
        self.resblock0_1 = nn.Sequential(*resblock0_1)
        self.resblock1_1 = nn.Sequential(*resblock1_1)
        self.resblock2_1 = nn.Sequential(*resblock2_1)
        self.model_out1 = nn.Sequential(*model_out1)
        self.model_out2 = nn.Sequential(*model_out2)
        self.model_out3 = nn.Sequential(*model_out3)

    def forward(self, input, bias_input, ref_input, ref_color):
        bias_input = bias_input.view(input.shape[0], -1, 1, 1)
        in_1 = self.model1(input)
        in_2 = self.model2(in_1[:,:,::2,::2])
        in_3 = self.model3(in_2[:,:,::2,::2])
        in_4 = self.model4(in_3[:,:,::2,::2])
        in_5 = self.model5(in_4)
        in_6 = self.model6(in_5)

        ref_1 = self.model1(ref_input)
        ref_2 = self.model2(ref_1[:,:,::2,::2])
        ref_3 = self.model3(ref_2[:,:,::2,::2])
        ref_4 = self.model4(ref_3[:,:,::2,::2])
        ref_5 = self.model5(ref_4)
        ref_6 = self.model6(ref_5)
        
        t1 = F.interpolate(in_1, scale_factor=0.5, mode='bilinear')
        t2 = in_2
        t3 = F.interpolate(in_3, scale_factor=2, mode='bilinear')
        t4 = F.interpolate(in_4, scale_factor=4, mode='bilinear')
        t5 = F.interpolate(in_5, scale_factor=4, mode='bilinear')
        t6 = F.interpolate(in_6, scale_factor=4, mode='bilinear')
        t = torch.cat((t1, t2, t3, t4, t5, t6), dim=1)

        r1 = F.interpolate(ref_1, scale_factor=0.5, mode='bilinear')
        r2 = ref_2
        r3 = F.interpolate(ref_3, scale_factor=2, mode='bilinear')
        r4 = F.interpolate(ref_4, scale_factor=4, mode='bilinear')
        r5 = F.interpolate(ref_5, scale_factor=4, mode='bilinear')
        r6 = F.interpolate(ref_6, scale_factor=4, mode='bilinear')
        r = torch.cat((r1, r2, r3, r4, r5, r6), dim=1)

        input_T_flatten = t.view(t.shape[0], t.shape[1], -1).permute(0, 2, 1)
        input_R_flatten = r.view(r.shape[0], r.shape[1], -1).permute(0, 2, 1)
        input_T_flatten = input_T_flatten / torch.norm(input_T_flatten, p=2, dim=-1, keepdim=True)
        input_R_flatten = input_R_flatten / torch.norm(input_R_flatten, p=2, dim=-1, keepdim=True)
        corr = torch.bmm(input_R_flatten, input_T_flatten.permute(0, 2, 1))

        conf = self.conf_feature(corr)
        conf = conf.view(conf.shape[0], 1, t2.shape[2], t2.shape[3])
        conf_1 = conf
        conf_2 = conf_1[:,:,::2,::2]
        conf_3 = conf_2[:,:,::2,::2]
        
        corr = F.softmax(corr/0.01, dim=1)
        align_1, align_2, align_3 = self.ref_network(ref_color, corr, t2.shape[2], t2.shape[3])
        conv_global1, conv_global2, conv_global3 = self.global_network(bias_input)

        conv1_2 = self.model1(input)
        conv2_2 = self.model2(conv1_2[:,:,::2,::2])
        conv3_3 = self.model3(conv2_2[:,:,::2,::2])
        conv4_3 = self.model4(conv3_3[:,:,::2,::2])
        conv5_3 = self.model5(conv4_3)
        conv6_3 = self.model6(conv5_3)

        class_output = self.classify_network(conv6_3)

        conv_global1_repeat = conv_global1.expand_as(conv6_3)
        conv6_3_global = conv6_3 + align_3 * conf_3 + conv_global1_repeat * (1 - conf_3)
        conv7_resblock1 = self.resblock0_1(conv6_3_global)
        conv7_resblock2 = self.resblock0_2(conv7_resblock1)
        conv7_resblock3 = self.resblock0_3(conv7_resblock2)
        conv8_up = self.model8up(conv7_resblock3) + self.model3short8(conv3_3)
        conv8_3 = self.model8(conv8_up)
        fake_img1 = self.model_out1(conv8_3)

        conv_global2_repeat = conv_global2.expand_as(conv8_3)
        conv8_3_global = conv8_3 + align_2 * conf_2 + conv_global2_repeat * (1 - conf_2)
        conv8_resblock1 = self.resblock1_1(conv8_3_global)
        conv8_resblock2 = self.resblock1_2(conv8_resblock1)
        conv8_resblock3 = self.resblock1_3(conv8_resblock2)
        conv9_up = self.model9up(conv8_resblock3) + self.model2short9(conv2_2)
        conv9_3 = self.model9(conv9_up)
        fake_img2 = self.model_out2(conv9_3)

        conv_global3_repeat = conv_global3.expand_as(conv9_3)
        conv9_3_global = conv9_3 + align_1 * conf_1 + conv_global3_repeat * (1 - conf_1)
        conv9_resblock1 = self.resblock2_1(conv9_3_global)
        conv9_resblock2 = self.resblock2_2(conv9_resblock1)
        conv9_resblock3 = self.resblock2_3(conv9_resblock2)
        conv10_up = self.model10up(conv9_resblock3) + self.model1short10(conv1_2)
        conv10_2 = self.model10(conv10_up)
        fake_img3 = self.model_out3(conv10_2)

        return [fake_img1, fake_img2, fake_img3], class_output, conf
