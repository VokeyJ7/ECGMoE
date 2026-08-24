import torch
import torch.nn as nn


class ECGFeatureExtractor(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, conv_stride):
    super().__init__()
    padding = (kernel_size - 1) // 2
    self.net = nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=conv_stride, padding=padding),
                             nn.SiLU(),
                             nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=conv_stride, padding=padding),
                             nn.SiLU()
                            )
    self.gap = nn.AdaptiveAvgPool1d(2)
    self.flatten = nn.Flatten(1)

  def forward(self, x):
    latent_vector = self.net(x) 
    latent_vector = self.gap(latent_vector)
    latent_vector = self.flatten(latent_vector) 
    return latent_vector


class Router(nn.Module):
  def __init__(self, num_experts):
    super().__init__()
    self.net = nn.LazyLinear(num_experts)

  def forward(self, latent_vector):
    x = self.net(latent_vector)
    router_weights = nn.functional.softmax(x, dim=-1) 

    return router_weights


class Expert(nn.Module):
  def __init__(self, output_dims, h1):
    super().__init__()
    self.net = nn.Sequential(
        nn.LazyLinear(h1),
        nn.SiLU(),
        nn.Linear(h1, output_dims)

    )

  def forward(self, x):
    x = self.net(x)
    return x

class Classifier(nn.Module):
  def __init__(self, class_num):
    super().__init__()
    self.net = nn.LazyLinear(class_num)

  def forward(self, x):
    return self.net(x)

class MoE(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, conv_stride, output_dims, h1, num_experts, class_num):
    super().__init__()

    self.featureExtractor = ECGFeatureExtractor(in_channels, out_channels, kernel_size, conv_stride)
    self.router = Router(num_experts)
    self.experts = nn.ModuleList([Expert(output_dims, h1) for _ in range(num_experts)])
    self.classifier = Classifier(class_num)

  def forward(self, x):
    latent_vector = self.featureExtractor(x)
    router_weights = self.router(latent_vector)


    router_weights = router_weights.unsqueeze(-1)

    expert_outputs = []

    for expert in self.experts:
      expert_outputs.append(expert(latent_vector))


    expert_outputs = torch.stack(expert_outputs, dim=1) 


    weighted_combination = (expert_outputs * router_weights).sum(dim=1)


    out = self.classifier(weighted_combination)

    return out, weighted_combination, router_weights



