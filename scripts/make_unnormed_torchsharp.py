import torch
import exportsd

# Load the original state_dict
state_dict = torch.load('pytorch_model.bin')

# For each weight_norm layer, reconstruct the actual weights
for key in list(state_dict.keys()):
    if 'parametrizations.weight.original0' in key:
        # Get the base key (e.g., 'encoder.block.0')
        base_key = key.split('.parametrizations')[0]
        
        # Compute the actual weight
        g = state_dict[f"{base_key}.parametrizations.weight.original0"]  # magnitude (1,)
        v = state_dict[f"{base_key}.parametrizations.weight.original1"]  # direction
        weight = v * (g / torch.norm(v, dim=(1,2), keepdim=True))
        
        # Store as regular weight
        state_dict[f"{base_key}.weight"] = weight
        
        # Remove the parameterized versions
        del state_dict[key]
        del state_dict[f"{base_key}.parametrizations.weight.original1"]

torch.save(state_dict, "pytorch_model_unnormed.bin")