import torch
import utility
import data
import model
import loss
from torchsummary import summary
import torchstat
from option import args
from trainer import Trainer
from torch.nn.utils import parameters_to_vector
from ptflops import get_model_complexity_info
import re
from torch.autograd import profiler

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def count_operations(model, input_size=(3, 192, 192), idx_scale=0):
    input = torch.randn(1, *input_size).to(next(model.parameters()).device)
    flops_count = [0]
    mult_count = [0]
    add_count = [0]
    mac_count = [0]
    conv_count = [0]

    def conv2d_ops(module, input, output):
        batch_size = input[0].size(0)
        input_channels = module.in_channels
        output_channels = module.out_channels
        kernel_height, kernel_width = module.kernel_size
        out_height, out_width = output.size()[-2:]
        
        # MACs: kernel_size^2 * in_channels * out_channels * out_height * out_width
        macs = kernel_height * kernel_width * input_channels * output_channels * out_height * out_width
        
        # Multiplications are the same as MACs
        mults = macs
        
        # Additions: kernel_size^2 * in_channels * out_channels * out_height * out_width
        # (one less than MACs per output element)
        adds = macs - output_channels * out_height * out_width
        
        if module.bias is not None:
            adds += output_channels * out_height * out_width
        
        flops_count[0] += mults + adds
        mult_count[0] += mults
        add_count[0] += adds
        mac_count[0] += macs
        conv_count[0] += 1

    def linear_ops(module, input, output):
        batch_size = input[0].size(0)
        macs = batch_size * module.in_features * module.out_features
        mults = macs
        adds = macs - batch_size * module.out_features
        
        if module.bias is not None:
            adds += batch_size * module.out_features
        
        flops_count[0] += mults + adds
        mult_count[0] += mults
        add_count[0] += adds
        mac_count[0] += macs

    def register_hooks(module):
        if isinstance(module, torch.nn.Conv2d):
            module.register_forward_hook(conv2d_ops)
        if isinstance(module, torch.nn.Linear):
            module.register_forward_hook(linear_ops)
    
    model.apply(register_hooks)
    
    with torch.no_grad():
        model(input, idx_scale)
    
    return {
        'flops': flops_count[0],
        'multiplications': mult_count[0],
        'additions': add_count[0],
        'macs': mac_count[0],
        'convolutions': conv_count[0]
    }

def main():
    global model
    if args.data_test == 'video':
        from videotester import VideoTester
        model = model.Model(args, checkpoint)
        t = VideoTester(args, model, checkpoint)
        t.test()
    else:
        if checkpoint.ok:
            loader = data.Data(args)
            _model = model.Model(args, checkpoint)
            _loss = loss.Loss(args, checkpoint) if not args.test_only else None
            t = Trainer(args, loader, _model, _loss, checkpoint)
            
            # Count parameters
            num_params = count_parameters(_model)
            print(f'The number of parameters: {num_params:,}')

            # Count operations
            ops = count_operations(_model)
            print(f"Estimated FLOPs: {ops['flops']:,}")
            print(f"Estimated Multiplications: {ops['multiplications']:,}")
            print(f"Estimated Additions: {ops['additions']:,}")
            print(f"Estimated Multiply-Accumulates (MACs): {ops['macs']:,}")
            print(f"Number of Convolutions: {ops['convolutions']:,}")

            # Training loop
            while not t.terminate():
                t.train()
                t.test()

            checkpoint.done()

if __name__ == '__main__':
    main()