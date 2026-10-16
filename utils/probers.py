import torch, os

class BaseProber:
    def __init__(self, model, mode, base_dir, direction_type=None, heads=False, safe_direction_dir=None, unsafe_direction_dir=None, target_layer_idx=None, save_full=False, **kwargs):
        self.model = model
        self.heads = heads
        self.mode = mode
        self.direction_type = direction_type
        assert self.mode in ['collect', 'apply'], "Mode must be either 'collect' or 'apply'"
        self.base_dir = base_dir
        self.safe_direction_dir = safe_direction_dir
        self.unsafe_direction_dir = unsafe_direction_dir
        self.activations = {}
        self.tmp_activations = {}
        self.hooks = []
        self.target_layer_idx = target_layer_idx
        self.kwargs = kwargs
        self.save_full = save_full
        if isinstance(self.target_layer_idx, int):
            self.target_layer_idx = [self.target_layer_idx]
        self.init_hooks()

    def init_hooks(self):
        raise NotImplementedError

    def get_activations(self):
        return self.activations

    def load_tensor(self, direction_path):
        return torch.load(direction_path, map_location='cpu', weights_only=False)

class MARS(BaseProber):
    def __init__(self, collect_type=None, act_mean_path=None, n_images_mean=None, act_mean_type=None, target_token_id=-1, **kwargs):
        self.collect_type = collect_type
        self.act_mean_path = act_mean_path
        self.n_images_mean = n_images_mean
        self.act_mean_type = act_mean_type
        self.target_token_id = target_token_id
        super().__init__(**kwargs)

    def get_blocks(self):
        if hasattr(self.model, 'layers'):
            return self.model.layers
        elif hasattr(self.model, 'blocks'):
            return self.model.blocks
        else:
            raise NotImplementedError('Not implemented for this decoder. Please check the how to collect the layers.')

    def init_hooks(self):
        blocks = self.get_blocks()
        for layer_idx, block in enumerate(blocks):
            block.layer_idx = layer_idx
            if self.mode == 'collect':
                if self.collect_type == 'text':
                    block.register_forward_hook(self.cache_tmp_activations)
                elif self.collect_type == 'image':
                    block.register_forward_hook(self.cache_full_activations)

            elif self.mode == 'apply':
                if self.target_layer_idx is None or (layer_idx in self.target_layer_idx):
                    accept_act = self.load_tensor(
                        os.path.join(self.safe_direction_dir, str(layer_idx), 'output_direction_full.pt')
                    )
                    reject_act = self.load_tensor(
                        os.path.join(self.unsafe_direction_dir, str(layer_idx), 'output_direction_full.pt')
                    )

                    accept_centroid = accept_act.mean(dim=0)
                    reject_centroid = reject_act.mean(dim=0)
                    txt_mean = torch.cat([accept_act, reject_act], dim=0).mean(dim=0)
                    block.accept_centroid_centered = (accept_centroid - txt_mean).to(self.model.device)
                    block.reject_centroid_centered = (reject_centroid - txt_mean).to(self.model.device)

                    image_mean_path = os.path.join(
                        self.act_mean_path,
                        str(layer_idx),
                        f"output_act_{self.act_mean_type}_{str(self.n_images_mean)}_mean.pt"
                    )

                    block.image_mean = self.load_tensor(image_mean_path).to(self.model.device)

                    direction = reject_centroid - accept_centroid
                    if self.direction_type == 'mars':
                        block.direction = (direction / (direction.norm(dim=-1, keepdim=True)+1e-7)).to(self.model.device)
                        self.hooks.append(
                            block.register_forward_hook(self.mars)
                        )
                    elif self.direction_type == 'ablation_no_relu':
                        block.direction = (direction / (direction.norm(dim=-1, keepdim=True)+1e-7)).to(self.model.device)
                        self.hooks.append(
                            block.register_forward_hook(self.ablation_no_relu)
                        )
                    elif self.direction_type == 'ablation_no_centering':
                        block.accept_centroid = accept_centroid.to(self.model.device)
                        block.reject_centroid = reject_centroid.to(self.model.device)

                        block.direction = (direction / (direction.norm(dim=-1, keepdim=True)+1e-7)).to(self.model.device)
                        self.hooks.append(
                            block.register_forward_hook(self.ablation_no_centering)
                        )
                    else:
                        raise NotImplementedError('This direction is not implemented.')

    # MARS
    def mars(self, module, input, output):
        if isinstance(output, tuple):
            is_tuple = True
            output_tensor = output[0]
        else:
            is_tuple = False
            output_tensor = output

        if output_tensor.shape[1] == 1:
            return output

        target_idx = self.target_token_id
        output_centered = output_tensor[:, target_idx] - module.image_mean[target_idx]        

        dist_accept = torch.norm(output_centered - module.accept_centroid_centered, dim=-1)  # [B, T]
        dist_reject = torch.norm(output_centered - module.reject_centroid_centered, dim=-1)  # [B, T]

        coeff = torch.einsum('...d,d->...', output_centered, module.direction).unsqueeze(-1)
        delta = torch.relu(coeff) * module.direction                                          # Eq. 5 of the paper

        use_accept = (dist_accept < dist_reject)
        radius = torch.where(use_accept, dist_accept, dist_reject)                            # Eq. 6 of the paper
        max_alpha = radius / ((coeff * module.direction).norm(dim=-1)+1e-07)                  # Eq. 7 of the paper

        if not is_tuple:
            return output + ((max_alpha * self.kwargs['coeff_alpha']).unsqueeze(-1) * delta).unsqueeze(dim=1)
        else:
            return (output[0] + ((max_alpha * self.kwargs['coeff_alpha']).unsqueeze(-1) * delta).unsqueeze(dim=1), )

    # Ablation: w/o ReLU gating
    def ablation_no_relu(self, module, input, output):
        if isinstance(output, tuple):
            is_tuple = True
            output_tensor = output[0]
        else:
            is_tuple = False
            output_tensor = output

        if output_tensor.shape[1] == 1:
            return output
        output_centered = output_tensor[:, self.target_token_id] - module.image_mean[self.target_token_id]

        dist_accept = torch.norm(output_centered - module.accept_centroid_centered, dim=-1)  # [B, T]
        dist_reject = torch.norm(output_centered - module.reject_centroid_centered, dim=-1)  # [B, T]

        use_accept = (dist_accept < dist_reject)
        radius = torch.where(use_accept, dist_accept, dist_reject)

        delta = torch.einsum('...d,d->...', output_centered, module.direction).unsqueeze(-1) * module.direction
        max_alpha = radius / (delta.norm(dim=-1)+1e-07)

        if not is_tuple:
            return output + ((max_alpha * self.kwargs['coeff_alpha']).unsqueeze(-1) * delta).unsqueeze(dim=1)
        else:
            return (output[0] + ((max_alpha * self.kwargs['coeff_alpha']).unsqueeze(-1) * delta).unsqueeze(dim=1), )

    # Ablation: w/o centering
    def ablation_no_centering(self, module, input, output):
        output_tensor = output[0] if (is_tuple := isinstance(output, tuple)) else output

        if output_tensor.shape[1] == 1:
            return output

        radius = torch.minimum(
            torch.norm(output_tensor[:, self.target_token_id] - module.accept_centroid, dim=-1), # dist accept
            torch.norm(output_tensor[:, self.target_token_id] - module.reject_centroid, dim=-1) # dist reject
        )
        coeff     = (output_tensor[:, self.target_token_id] @ module.direction).unsqueeze(-1)
        delta     = torch.relu(coeff) * module.direction
        max_alpha = radius / ((coeff * module.direction).norm(dim=-1) + 1e-7)

        return (output_tensor + ((max_alpha * self.kwargs['coeff_alpha']).unsqueeze(-1) * delta).unsqueeze(dim=1),) if is_tuple else output_tensor + ((max_alpha * self.kwargs['coeff_alpha']).unsqueeze(-1) * delta).unsqueeze(dim=1)

    # Text directions: cache the target token of the current prompt, kept only if the response matches the expected behavior
    def cache_tmp_activations(self, module, input, output):
        if isinstance(output, tuple):
            output_tensor = output[0]
        else:
            output_tensor = output
        if output_tensor.shape[-2] != 1:
            if hasattr(self.model, 'refusal_at_token_idx'):
                curr_token = self.model.refusal_at_token_idx
            else:
                raise NotImplementedError('Not implemented yet.')

            input_decomposed = input[0][:, curr_token, :].squeeze(dim=1).cpu().detach()
            output_decomposed = output_tensor[:, curr_token, :].squeeze(dim=1).cpu().detach()

            self.tmp_activations[module.layer_idx] = {
                'input': input_decomposed,
                'output': output_decomposed
            }

        return output

    def clear_tmp(self):
        self.tmp_activations = {}

    def accept_forward(self):
        blocks = self.get_blocks()
        for layer_idx in range(len(blocks)):
            if layer_idx not in self.activations:
                self.activations[layer_idx] = {
                    'input': self.tmp_activations[layer_idx]['input'],
                    'output': self.tmp_activations[layer_idx]['output']
                }
            else:
                self.activations[layer_idx] = {
                    'input': torch.cat((self.activations[layer_idx]['input'], self.tmp_activations[layer_idx]['input']), dim=0),
                    'output': torch.cat((self.activations[layer_idx]['output'], self.tmp_activations[layer_idx]['output']), dim=0)
                }

    # Neutral images: full-sequence activations of every image
    def cache_full_activations(self, module, input, output):
        if isinstance(output, tuple):
            output_tensor = output[0]
        else:
            output_tensor = output

        output_decomposed = output_tensor.cpu().detach()

        if module.layer_idx not in self.activations:
            self.activations[module.layer_idx] = {
                'output': output_decomposed
            }
        else:
            self.activations[module.layer_idx] = {
                'output': torch.cat((self.activations[module.layer_idx]['output'], output_decomposed), dim=0)
            }
        return output

    def save_img_activations(self, image_type, n_images):
        activations = self.get_activations()
        for layer_idx in activations:
            save_path = os.path.join(self.base_dir, str(layer_idx))
            os.makedirs(save_path, exist_ok=True)
            output_act = activations[layer_idx]['output'].mean(dim=0)
            torch.save(output_act, os.path.join(save_path, f'output_act_{image_type}_{n_images}_mean.pt'), pickle_protocol=4)

    def save_activations(self):
        activations = self.get_activations()
        for layer_idx in activations:
            save_path = os.path.join(self.base_dir, str(layer_idx))
            os.makedirs(save_path, exist_ok=True)
            output_direction = activations[layer_idx]['output'].mean(dim=0)
            torch.save(output_direction, os.path.join(save_path, 'output_direction.pt'), pickle_protocol=4)

            if self.save_full:
                torch.save(activations[layer_idx]['output'], os.path.join(save_path, 'output_direction_full.pt'), pickle_protocol=4)
