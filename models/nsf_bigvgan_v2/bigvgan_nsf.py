# Copyright (c) 2024 NVIDIA CORPORATION.
# Licensed under the MIT license.
# Adapted for DiffSinger integration with NSF (Neural Source Filter) for Singing Voice Synthesis.

import torch
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import weight_norm, remove_weight_norm
import numpy as np

# Importações do BigVGAN V2 original
from . import activations
from .utils import init_weights, get_padding
from .alias_free_activation.torch.act import Activation1d as TorchActivation1d
from .env import AttrDict

# Importação do módulo NSF (Neural Source Filter) do seu repositório original
from .nsf import SourceModuleHnNSF


class AMPBlock1(torch.nn.Module):
    """
    AMPBlock1 do BigVGAN V2.
    Aplica ativações SnakeBeta com anti-aliasing.
    """
    def __init__(
        self,
        h: AttrDict,
        channels: int,
        kernel_size: int = 3,
        dilation: tuple = (1, 3, 5),
        activation: str = None,
    ):
        super().__init__()
        self.h = h
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        stride=1,
                        dilation=d,
                        padding=get_padding(kernel_size, d),
                    )
                )
                for d in dilation
            ]
        )
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        stride=1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                )
                for _ in range(len(dilation))
            ]
        )
        self.convs2.apply(init_weights)

        self.num_layers = len(self.convs1) + len(self.convs2)

        # Lazy-load da versão CUDA se disponível (para inferência rápida)
        if self.h.get("use_cuda_kernel", False):
            from .alias_free_activation.cuda.activation1d import (
                Activation1d as CudaActivation1d,
            )
            Activation1d = CudaActivation1d
        else:
            Activation1d = TorchActivation1d

        # Seleção da função de ativação
        if activation == "snake":
            self.activations = nn.ModuleList(
                [
                    Activation1d(
                        activation=activations.Snake(
                            channels, alpha_logscale=h.snake_logscale
                        )
                    )
                    for _ in range(self.num_layers)
                ]
            )
        elif activation == "snakebeta":
            self.activations = nn.ModuleList(
                [
                    Activation1d(
                        activation=activations.SnakeBeta(
                            channels, alpha_logscale=h.snake_logscale
                        )
                    )
                    for _ in range(self.num_layers)
                ]
            )
        else:
            raise NotImplementedError(
                "activation incorrectly specified. check the config file and look for 'activation'."
            )

    def forward(self, x):
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2):
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class BigVGAN_NSF(torch.nn.Module):
    """
    BigVGAN V2 Híbrido com injeção de F0 via NSF (Neural Source Filter).
    Combina a arquitetura state-of-the-art do BigVGAN V2 (anti-aliasing, SnakeBeta)
    com a precisão cirúrgica de pitch do NSF para Síntese de Voz Cantada.
    """
    def __init__(self, h: AttrDict, use_cuda_kernel: bool = False):
        super().__init__()
        self.h = h
        self.h["use_cuda_kernel"] = use_cuda_kernel

        # Seleção da Activation1d (CUDA ou PyTorch puro)
        if self.h.get("use_cuda_kernel", False):
            from .alias_free_activation.cuda.activation1d import (
                Activation1d as CudaActivation1d,
            )
            Activation1d = CudaActivation1d
        else:
            Activation1d = TorchActivation1d

        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)

        # ==============================
        # MÓDULO NSF (Neural Source Filter)
        # ==============================
        # Gera a fonte harmônica a partir do F0
        self.m_source = SourceModuleHnNSF(
            sampling_rate=h.sampling_rate,
            harmonic_num=8,  # 8 harmônicos acima do F0
            sine_amp=0.1,
            add_noise_std=0.003,
            voiced_threshold=0
        )
        
        # Upsample do F0 para a resolução do áudio
        self.f0_upsamp = torch.nn.Upsample(
            scale_factor=np.prod(h.upsample_rates)
        )
        
        # Convoluções para injetar a fonte harmônica em cada etapa de upsampling
        self.noise_convs = nn.ModuleList()
        for i, u in enumerate(h.upsample_rates):
            if i + 1 < len(h.upsample_rates):
                stride_f0 = np.prod(h.upsample_rates[i + 1:])
                stride_f0 = int(stride_f0)
                self.noise_convs.append(
                    Conv1d(
                        1,
                        h.upsample_initial_channel // (2 ** (i + 1)),
                        kernel_size=stride_f0 * 2,
                        stride=stride_f0,
                        padding=stride_f0 // 2,
                    )
                )
            else:
                self.noise_convs.append(
                    Conv1d(
                        1,
                        h.upsample_initial_channel // (2 ** (i + 1)),
                        kernel_size=1
                    )
                )

        # ==============================
        # ARQUITETURA BIGVGAN V2
        # ==============================
        # Pre-conv
        self.conv_pre = weight_norm(
            Conv1d(h.num_mels, h.upsample_initial_channel, 7, 1, padding=3)
        )

        # Transposed conv-based upsamplers
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(
                nn.ModuleList(
                    [
                        weight_norm(
                            ConvTranspose1d(
                                h.upsample_initial_channel // (2**i),
                                h.upsample_initial_channel // (2 ** (i + 1)),
                                k,
                                u,
                                padding=(k - u) // 2,
                            )
                        )
                    ]
                )
            )

        # Residual blocks usando AMPBlock1 (anti-aliased)
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)
            ):
                self.resblocks.append(
                    AMPBlock1(h, ch, k, d, activation=h.activation)
                )

        # Post-conv
        activation_post = (
            activations.Snake(ch, alpha_logscale=h.snake_logscale)
            if h.activation == "snake"
            else (
                activations.SnakeBeta(ch, alpha_logscale=h.snake_logscale)
                if h.activation == "snakebeta"
                else None
            )
        )
        if activation_post is None:
            raise NotImplementedError(
                "activation incorrectly specified. check the config file and look for 'activation'."
            )
        self.activation_post = Activation1d(activation=activation_post)

        self.use_bias_at_final = h.get("use_bias_at_final", False)
        self.conv_post = weight_norm(
            Conv1d(ch, 1, 7, 1, padding=3, bias=self.use_bias_at_final)
        )

        # Weight initialization
        for i in range(len(self.ups)):
            self.ups[i].apply(init_weights)
        self.conv_post.apply(init_weights)

        self.use_tanh_at_final = h.get("use_tanh_at_final", False)

    def forward(self, x, f0):
        """
        Forward pass do BigVGAN V2 com injeção de F0 via NSF.
        
        Args:
            x (Tensor): Espectrograma Mel de entrada (B, num_mels, T).
            f0 (Tensor): Frequência fundamental em Hz (B, T_f0).
        
        Returns:
            Tensor: Áudio gerado (B, 1, T_audio).
        """
        # ==============================
        # PROCESSAMENTO NSF (Fonte Harmônica)
        # ==============================
        # Upsample do F0 para a resolução do áudio
        f0_upsampled = self.f0_upsamp(f0.unsqueeze(1)).transpose(1, 2)  # (B, T_audio, 1)
        har_source = self.m_source(f0_upsampled)  # (B, T_audio, 1)
        har_source = har_source.transpose(1, 2)  # (B, 1, T_audio)

        # ==============================
        # PROCESSAMENTO BIGVGAN V2
        # ==============================
        # Pre-conv
        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            # Upsampling
            for i_up in range(len(self.ups[i])):
                x = self.ups[i][i_up](x)
            
            # INJEÇÃO NSF: Soma a fonte harmônica no canal atual
            x_source = self.noise_convs[i](har_source)
            x = x + x_source
            
            # AMP blocks (anti-aliased)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        # Post-conv
        x = self.activation_post(x)
        x = self.conv_post(x)

        # Final activation
        if self.use_tanh_at_final:
            x = torch.tanh(x)
        else:
            x = torch.clamp(x, min=-1.0, max=1.0)

        return x

    def remove_weight_norm(self):
        try:
            print("Removing weight norm...")
            for l in self.ups:
                for l_i in l:
                    remove_weight_norm(l_i)
            for l in self.resblocks:
                l.remove_weight_norm()
            remove_weight_norm(self.conv_pre)
            remove_weight_norm(self.conv_post)
        except ValueError:
            print("[INFO] Model already removed weight norm. Skipping!")
            pass
