import torch
import torch.nn as nn
from torch.autograd import Function
from torch import manual_seed


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, lamb):
        ctx.lamb = lamb
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lamb, None


class DANN(nn.Module):

    def __init__(self, config):
        manual_seed(config['random_seed'])

        super(DANN, self).__init__()
        self.cnn = nn.Sequential()
        self.cnn.add_module('f_conv1', nn.Conv2d(
            3, config['conv1_channels'],
            kernel_size=config['conv_kernel'], padding='same'
        ))
        self.cnn.add_module('f_bn1',   nn.BatchNorm2d(config['conv1_channels']))
        self.cnn.add_module('f_drop1', nn.Dropout2d(config['conv1_dropout']))
        self.cnn.add_module('f_pool1', nn.MaxPool2d(2))
        self.cnn.add_module('f_relu1', nn.ReLU(True))
        self.cnn.add_module('f_conv2', nn.Conv2d(
            config['conv1_channels'], config['conv2_channels'],
            kernel_size=config['conv_kernel'], padding='same'
        ))
        self.cnn.add_module('f_bn2',   nn.BatchNorm2d(config['conv2_channels']))
        self.cnn.add_module('f_drop2', nn.Dropout2d(config['conv2_dropout']))
        self.cnn.add_module('f_pool2', nn.MaxPool2d(2))
        self.cnn.add_module('f_relu2', nn.ReLU(True))

        self.rb_clf = nn.Sequential()
        self.rb_clf.add_module('c_fc1',   nn.Linear(config['conv2_channels'] *
                                                    (config['image_size']//4)**2,
                                                    config['class1_neurons']))
        self.rb_clf.add_module('c_bn1',   nn.BatchNorm1d(config['class1_neurons']))
        self.rb_clf.add_module('c_relu1', nn.ReLU(True))
        self.rb_clf.add_module('c_drop1', nn.Dropout(config['class1_dropout']))
        self.rb_clf.add_module('c_fc2',   nn.Linear(config['class1_neurons'],
                                                    config['class2_neurons']))
        self.rb_clf.add_module('c_bn2',   nn.BatchNorm1d(config['class2_neurons']))
        self.rb_clf.add_module('c_relu2', nn.ReLU(True))
        self.rb_clf.add_module('c_drop2', nn.Dropout(config['class2_dropout']))
        self.rb_clf.add_module('c_fc3',   nn.Linear(config['class2_neurons'],
                                                    config['class3_neurons']))
        self.rb_clf.add_module('c_bn3',   nn.BatchNorm1d(config['class3_neurons']))
        self.rb_clf.add_module('c_relu3', nn.ReLU(True))
        self.rb_clf.add_module('c_drop3', nn.Dropout(config['class3_dropout']))
        self.rb_clf.add_module('c_fc4',   nn.Linear(config['class3_neurons'], 1))
        self.rb_clf.add_module('c_soft',  nn.Sigmoid())

        self.domain_clf = nn.Sequential()
        self.domain_clf.add_module('d_fc1',   nn.Linear(config['conv2_channels'] *
                                                        (config['image_size']//4)**2,
                                                        config['domain1_neurons']))
        self.domain_clf.add_module('d_bn1',   nn.BatchNorm1d(config['domain1_neurons']))
        self.domain_clf.add_module('d_relu1', nn.ReLU(True))
        self.domain_clf.add_module('d_fc2',   nn.Linear(config['domain1_neurons'], 1))
        self.domain_clf.add_module('d_soft',  nn.Sigmoid())

    def forward(
        self,
        input_data: torch.Tensor,
        lamb: float = 1.0,
        include_domain_clf: bool = True
    ):
#     ) -> torch.Tuple[torch.Tensor, torch.Optional[torch.Tensor]]:

        image_size = input_data.data.shape[2]
        batch_size = input_data.data.shape[0]

        # only necessary when some inputs are 1 channel images
        input_data = input_data.expand(batch_size, 3, image_size, image_size)

        # Run one batch of images through feature extractor CNN
        feature = self.cnn(input_data)
        num_kernels = feature.shape[1]
        num_channels = (image_size // 4)**2
        feature = feature.view(-1, num_kernels * num_channels)

        rb_preds = self.rb_clf(feature).view(batch_size)

        # If in training mode, also run through domain classifier and return both scores
        if self.training:
            reverse_feature = ReverseLayerF.apply(feature, lamb)
            domain_preds = self.domain_clf(reverse_feature).view(batch_size)
            return rb_preds, domain_preds

        # When not in training mode, only return R/B score
        return rb_preds, None


def load_model():
    model = DANN()
    model.load_state_dict(torch.load('RBbot.pth', map_location=torch.device('cpu')))
    model.eval()
    return model


if __name__ == "__main__":
    model = load_model()
    input_tensor = torch.randn(1, 3, 31, 31)  # Example input
    output = model(input_tensor)
    print(output)
