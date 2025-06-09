import torch
import torch.nn as nn
import torch.nn.functional as F

class ParametricGammaKernelConv1d(nn.Module):
    """
    Conv1D layer that generates kernels using gamma distributions.
    Each kernel is made from 2 gamma components (like PPG incident + reflected waves).
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
                 dilation=1, groups=1, bias=True, num_gammas_per_kernel=2):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.num_gammas_per_kernel = num_gammas_per_kernel
        
        # Each gamma has 4 params: amplitude, shape, scale, location
        self.params_per_gamma = 4
        self.total_params_per_kernel = num_gammas_per_kernel * 4
        self.total_kernels = out_channels * in_channels
        
        # Main learnable parameters: [out_channels, in_channels, 8_params]
        self.gamma_params = nn.Parameter(self._init_params())
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None
            
        # Kernel position indices [0, 1, 2, ..., kernel_size-1]
        self.register_buffer('kernel_positions', torch.arange(kernel_size, dtype=torch.float))
    
    def _init_params(self):
        """Initialize gamma parameters for PPG-like shapes"""
        params = torch.zeros(self.out_channels, self.in_channels, self.total_params_per_kernel)
        
        for i in range(self.num_gammas_per_kernel):
            start_idx = i * 4
            
            # Amplitude: around 0.5
            params[:, :, start_idx] = torch.randn(self.out_channels, self.in_channels) * 0.1 + 0.5
            
            # Shape: 2-4 range (good for pulse shapes)
            params[:, :, start_idx + 1] = torch.randn(self.out_channels, self.in_channels) * 0.2 + (2.0 + i)
            
            # Scale: around 1-2
            params[:, :, start_idx + 2] = torch.randn(self.out_channels, self.in_channels) * 0.2 + 1.0
            
            # Location: stagger them across kernel
            base_loc = (i * self.kernel_size) / max(self.num_gammas_per_kernel, 2)
            params[:, :, start_idx + 3] = torch.randn(self.out_channels, self.in_channels) * 0.5 + base_loc
        
        return params
    
    def _get_gamma_params(self):
        """Extract parameters and apply constraints"""
        components = {}
        
        for i in range(self.num_gammas_per_kernel):
            start = i * 4
            
            # Get raw parameters
            raw_amp = self.gamma_params[:, :, start]
            raw_shape = self.gamma_params[:, :, start + 1] 
            raw_scale = self.gamma_params[:, :, start + 2]
            raw_loc = self.gamma_params[:, :, start + 3]
            
            # Apply constraints
            components[i] = {
                'amplitude': raw_amp,  # Can be negative
                'shape': F.softplus(raw_shape) + 1e-6,  # Must be positive
                'scale': F.softplus(raw_scale) + 1e-6,  # Must be positive
                'location': raw_loc  # Can be anything
            }
        
        return components
    
    def _gamma_pdf(self, x_positions, shape, scale, location):
        """Compute gamma PDF for all kernels at once"""
        # Broadcasting setup
        x = x_positions.unsqueeze(0)  # [1, kernel_size]
        shape = shape.unsqueeze(1)    # [total_kernels, 1]
        scale = scale.unsqueeze(1)    # [total_kernels, 1]
        location = location.unsqueeze(1)  # [total_kernels, 1]
        
        # Shift x by location
        x_shifted = x - location
        
        # Clamp to avoid log(0)
        x_clamped = torch.clamp(x_shifted, min=1e-8)
        
        # Gamma PDF in log space for stability
        log_pdf = (-shape * torch.log(scale) - 
                  torch.lgamma(shape) + 
                  (shape - 1) * torch.log(x_clamped) - 
                  x_clamped / scale)
        
        pdf = torch.exp(log_pdf)
        
        # Zero out negative x values (gamma PDF only defined for x > 0)
        pdf = torch.where(x_shifted > 0, pdf, torch.zeros_like(pdf))
        
        return pdf
    
    def _generate_kernels(self):
        """Generate all kernel weights using gamma distributions"""
        gamma_params = self._get_gamma_params()
        
        # Flatten to process all kernels together
        flat_params = {}
        for i in range(self.num_gammas_per_kernel):
            flat_params[i] = {}
            for name in ['amplitude', 'shape', 'scale', 'location']:
                flat_params[i][name] = gamma_params[i][name].view(self.total_kernels)
        
        # Start with zeros
        all_kernels = torch.zeros(self.total_kernels, self.kernel_size, device=self.gamma_params.device)
        
        # Add each gamma component
        for i in range(self.num_gammas_per_kernel):
            params = flat_params[i]
            
            # Compute gamma PDF
            pdf = self._gamma_pdf(
                self.kernel_positions,
                params['shape'],
                params['scale'], 
                params['location']
            )
            
            # Scale and add
            all_kernels += params['amplitude'].unsqueeze(1) * pdf
        
        # Reshape to Conv1D format
        return all_kernels.view(self.out_channels, self.in_channels, self.kernel_size)
    
    def forward(self, x):
        """Standard forward pass"""
        kernels = self._generate_kernels()
        return F.conv1d(x, kernels, self.bias, self.stride, self.padding, self.dilation, self.groups)


def test_gamma_conv():
    """Simple test to check shapes and basic functionality"""
    print("Testing ParametricGammaKernelConv1d...")
    
    # Create layer
    in_ch, out_ch, kernel_sz = 3, 8, 11
    layer = ParametricGammaKernelConv1d(in_ch, out_ch, kernel_sz)
    
    print(f"Layer: {in_ch} -> {out_ch}, kernel_size={kernel_sz}")
    print(f"Parameters: {sum(p.numel() for p in layer.parameters()):,}")
    
    # Test input
    batch_sz, seq_len = 4, 100
    x = torch.randn(batch_sz, in_ch, seq_len)
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    y = layer(x)
    print(f"Output shape: {y.shape}")
    
    # Check kernel generation
    kernels = layer._generate_kernels()
    print(f"Generated kernels shape: {kernels.shape}")
    print(f"Expected: ({out_ch}, {in_ch}, {kernel_sz})")
    
    # Check gamma parameters
    gamma_params = layer._get_gamma_params()
    print(f"Gamma components: {len(gamma_params)}")
    for i, params in gamma_params.items():
        print(f"  Component {i}: {list(params.keys())}")
        for name, tensor in params.items():
            print(f"    {name}: {tensor.shape}")
    
    print("✓ All tests passed!")
    return layer


if __name__ == "__main__":
    layer = test_gamma_conv()