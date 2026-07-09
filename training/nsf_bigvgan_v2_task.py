import pathlib
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchaudio
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from typing import Dict

# Imports do DiffSinger
from training.base_task_gan import GanBaseTask
from utils.wav2F0 import PITCH_EXTRACTORS_ID_TO_NAME, get_pitch
from utils.wav2mel import PitchAdjustableMelSpectrogram

# Imports do NSF-BigVGAN V2
from models.nsf_bigvgan_v2.bigvgan_nsf import BigVGAN_NSF
from models.nsf_bigvgan_v2.env import AttrDict
# Usamos o Discriminator do repo original (MRD + MPD + MSD)
from models.nsf_bigvgan_v2.discriminator import Discriminator
from modules.loss.bigvgan_v2_loss import BigVGANv2Loss


def spec_to_figure(spec, vmin=None, vmax=None):
    """Converte um espectrograma em uma figura matplotlib."""
    if isinstance(spec, torch.Tensor):
        spec = spec.cpu().numpy()
    fig = plt.figure(figsize=(12, 9), dpi=100)
    plt.pcolor(spec.T, vmin=vmin, vmax=vmax)
    plt.tight_layout()
    return fig


def dynamic_range_compression_torch(x, C=1, clip_val=1e-9):
    """Compressão de faixa dinâmica (log) para o Mel."""
    return torch.log(torch.clamp(x, min=clip_val) * C)


def wav_aug(wav, hop_size, speed=1):
    """Aumenta a velocidade do áudio (usado no Key Augmentation)."""
    orig_freq = int(np.round(hop_size * speed))
    new_freq = hop_size
    resample = torchaudio.transforms.Resample(
        orig_freq=orig_freq,
        new_freq=new_freq,
        lowpass_filter_width=128
    )
    wav_resampled = resample(wav)
    del resample
    return wav_resampled


def get_max_f0_from_config(config: dict):
    """Calcula o F0 máximo permitido com base na sample rate."""
    source_sr = config['audio_sample_rate']
    max_f0 = source_sr / 2
    return max_f0


class nsf_bigvgan_v2_dataset(Dataset):
    """
    Dataset para o NSF-BigVGAN V2.
    Lê arquivos .npz binarizados pelo DiffSinger com as chaves:
    - 'f0': Pitch em Hz (1D array)
    - 'mel': Espectrograma Mel (2D array, já normalizado com log)
    - 'audio': Áudio bruto (1D array, float32 entre -1 e 1)
    - 'pe': ID do pitch extractor usado (1=parselmouth, 2=harvest)
    """
    def __init__(self, config: dict, data_dir, infer=False):
        super().__init__()
        self.config = config
        self.data_dir = data_dir if isinstance(data_dir, pathlib.Path) else pathlib.Path(data_dir)
        
        # Lê o arquivo de índices (lista de caminhos para os .npz)
        with open(self.data_dir, 'r', encoding='utf8') as f:
            fills = f.read().strip().split('\n')
        self.data_index = fills
        self.infer = infer
        
        # Configurações de augmentação
        self.volume_aug = self.config.get('volume_aug', True)
        self.volume_aug_prob = self.config.get('volume_aug_prob', 0.5) if not infer else 0
        self.key_aug = self.config.get('key_aug', False)
        self.key_aug_prob = self.config.get('key_aug_prob', 0.5)
        
        # Se usar Key Augmentation, precisamos do MelSpectrogram para recalcular o Mel
        if self.key_aug:
            self.mel_spec_transform = PitchAdjustableMelSpectrogram(
                sample_rate=config['audio_sample_rate'],
                n_fft=config['fft_size'],
                win_length=config['win_size'],
                hop_length=config['hop_size'],
                f_min=config['fmin'],
                f_max=config['fmax'],
                n_mels=config['audio_num_mel_bins'],
            )
        
        self.max_f0 = get_max_f0_from_config(config)

    def __getitem__(self, index):
        sample = self.get_data(index)
        # Evita F0 acima do limite de Nyquist
        if sample['f0'].max() >= self.max_f0:
            return self.__getitem__(random.randint(0, len(self) - 1))
        return sample

    def __len__(self):
        return len(self.data_index)

    def get_data(self, index):
        data_path = pathlib.Path(self.data_index[index])
        data = np.load(data_path)
        pe_name = PITCH_EXTRACTORS_ID_TO_NAME[int(data['pe'])]
        
        # Se não estiver usando Key Augmentation, retorna os dados originais
        if self.infer or not self.key_aug or random.random() > self.key_aug_prob:
            return {
                'f0': data['f0'],
                'spectrogram': data['mel'],
                'audio': data['audio']
            }
        
        # Key Augmentation: muda a velocidade do áudio e recalcula Mel/F0
        speed = random.uniform(self.config.get('aug_min', 0.9), self.config.get('aug_max', 1.4))
        crop_mel_frames = int(np.ceil((self.config['crop_mel_frames'] + 4) * speed))
        samples_per_frame = self.config['hop_size']
        crop_wav_samples = crop_mel_frames * samples_per_frame
        
        if crop_wav_samples >= data['audio'].shape[0]:
            return {
                'f0': data['f0'],
                'spectrogram': data['mel'],
                'audio': data['audio']
            }
        
        start = random.randint(0, data['audio'].shape[0] - 1 - crop_wav_samples)
        end = start + crop_wav_samples
        audio = data['audio'][start:end]
        
        # Aumenta a velocidade do áudio
        audio_aug = wav_aug(torch.from_numpy(audio), self.config["hop_size"], speed=speed)
        
        # Recalcula o Mel com a nova velocidade
        mel_aug = dynamic_range_compression_torch(self.mel_spec_transform(audio_aug[None, :]))
        
        # Recalcula o F0 com a nova velocidade
        f0, uv = get_pitch(
            pe_name, audio, length=mel_aug.shape[-1], hparams=self.config,
            speed=speed, interp_uv=True
        )
        
        if f0 is None:
            return {
                'f0': data['f0'],
                'spectrogram': data['mel'],
                'audio': data['audio']
            }
        
        # Remove as bordas (2 frames de cada lado) para evitar artefatos
        audio_aug = audio_aug[2 * samples_per_frame: -2 * samples_per_frame].numpy()
        mel_aug = mel_aug[0, :, 2:-2].T.numpy()
        f0_aug = f0[2:-2] * speed
        
        return {
            'f0': f0_aug,
            'spectrogram': mel_aug,
            'audio': audio_aug
        }

    def collater(self, minibatch):
        """Empilha os samples em um batch, aplicando random cropping e volume augmentation."""
        samples_per_frame = self.config['hop_size']
        
        if self.infer:
            crop_mel_frames = 0
        else:
            crop_mel_frames = self.config['crop_mel_frames']
        
        for record in minibatch:
            # Filtra records muito curtos
            if record['spectrogram'].shape[0] < crop_mel_frames:
                del record['spectrogram']
                del record['audio']
                del record['f0']
                continue
            elif record['spectrogram'].shape[0] == crop_mel_frames:
                start = 0
            else:
                start = random.randint(0, record['spectrogram'].shape[0] - 1 - crop_mel_frames)
            
            end = start + crop_mel_frames
            
            if self.infer:
                record['spectrogram'] = record['spectrogram'].T
                record['f0'] = record['f0']
            else:
                record['spectrogram'] = record['spectrogram'][start:end].T
                record['f0'] = record['f0'][start:end]
            
            start *= samples_per_frame
            end *= samples_per_frame
            
            if self.infer:
                cty = (len(record['spectrogram'].T) * samples_per_frame)
                record['audio'] = record['audio'][:cty]
                record['audio'] = np.pad(
                    record['audio'],
                    (0, (len(record['spectrogram'].T) * samples_per_frame) - len(record['audio'])),
                    mode='constant'
                )
            else:
                record['audio'] = record['audio'][start:end]
                record['audio'] = np.pad(
                    record['audio'],
                    (0, (end - start) - len(record['audio'])),
                    mode='constant'
                )
        
        # Volume Augmentation
        if self.volume_aug:
            for record in minibatch:
                if record.get('audio') is None:
                    continue
                audio = record['audio']
                audio_mel = record['spectrogram']
                if random.random() < self.volume_aug_prob:
                    max_amp = float(np.max(np.abs(audio))) + 1e-5
                    max_shift = min(3, np.log(1 / max_amp))
                    log_mel_shift = random.uniform(-3, max_shift)
                    audio *= np.exp(log_mel_shift)
                    audio_mel += log_mel_shift
                audio_mel = torch.clamp(torch.from_numpy(audio_mel), min=np.log(1e-5)).numpy()
                record['audio'] = audio
                record['spectrogram'] = audio_mel
        
        # Empilha os tensores
        audio = np.stack([record['audio'] for record in minibatch if 'audio' in record])
        spectrogram = np.stack([record['spectrogram'] for record in minibatch if 'spectrogram' in record])
        f0 = np.stack([record['f0'] for record in minibatch if 'f0' in record])
        
        return {
            'audio': torch.from_numpy(audio).unsqueeze(1),  # [B, 1, T]
            'mel': torch.from_numpy(spectrogram),            # [B, n_mels, T_frames]
            'f0': torch.from_numpy(f0),                      # [B, T_frames]
        }


class nsf_bigvgan_v2_task(GanBaseTask):
    """
    Task do PyTorch Lightning para treinar o NSF-BigVGAN V2 no ecossistema DiffSinger.
    Herda de GanBaseTask, que já implementa:
    - Loop de treinamento manual (automatic_optimization = False)
    - Suporte a DDP (DistributedDataParallel)
    - Salvamento de checkpoints
    - Logging no TensorBoard
    - Suporte a fine-tuning e freezing de parâmetros
    """
    def __init__(self, config: dict, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        
        # MelSpectrogram para validação e PC-Augmentation
        self.TF = PitchAdjustableMelSpectrogram(
            sample_rate=config['audio_sample_rate'],
            n_fft=config['fft_size'],
            win_length=config['win_size'],
            hop_length=config['hop_size'],
            f_min=config['fmin'],
            f_max=config['fmax'],
            n_mels=config['audio_num_mel_bins'],
        )
        
        # Configurações de PC-Augmentation (Pitch Correction)
        self.pc_aug = self.config.get('pc_aug', False)
        self.pc_aug_rate = self.config.get('pc_aug_rate', 0.5)
        self.pc_aug_key = self.config.get('pc_aug_key', 5)
        
        # Para logar o áudio GT apenas uma vez por batch_idx
        self.logged_gt_wav = set()
        
        # Limite máximo de F0
        self.max_f0 = get_max_f0_from_config(config)

    def build_dataset(self):
        """Cria os datasets de treino e validação."""
        self.train_dataset = nsf_bigvgan_v2_dataset(
            config=self.config,
            data_dir=pathlib.Path(self.config['DataIndexPath']) / self.config['train_set_name']
        )
        self.valid_dataset = nsf_bigvgan_v2_dataset(
            config=self.config,
            data_dir=pathlib.Path(self.config['DataIndexPath']) / self.config['valid_set_name'],
            infer=True
        )

    def build_model(self):
        """Instancia o Gerador Híbrido (BigVGAN V2 + NSF) e os Discriminadores."""
        cfg = self.config['model_args'].copy()
        cfg.update({
            'sampling_rate': self.config['audio_sample_rate'],
            'num_mels': self.config['audio_num_mel_bins'],
            'hop_size': self.config['hop_size'],
            'n_fft': self.config['fft_size'],
            'win_size': self.config['win_size'],
            'fmin': self.config['fmin'],
            'fmax': self.config['fmax'],
        })
        h = AttrDict(cfg)
        
        # Instancia o Gerador Híbrido (BigVGAN V2 + NSF)
        # use_cuda_kernel=False para treino (CUDA kernel é só para inferência)
        self.generator = BigVGAN_NSF(h, use_cuda_kernel=False)
        
        # Instancia os Discriminadores (MRD + MPD + MSD)
        # Usamos o Discriminator do repo original que já empacota os 3
        self.discriminator = Discriminator(h)
        
        # === CARREGAMENTO DO PRÉ-TREINO DA NVIDIA (OPCIONAL) ===
        pretrain_path = self.config.get('nvidia_pretrain_path', None)
        if pretrain_path and pathlib.Path(pretrain_path).exists():
            self.load_nvidia_bigvgan_v2_pretrain(pretrain_path)

    def load_nvidia_bigvgan_v2_pretrain(self, checkpoint_path):
        """
        Carrega os pesos do BigVGAN V2 oficial da NVIDIA e prepara o modelo
        para o fine-tuning com NSF (Neural Source Filter).
        
        O "truque de ouro": inicializa as noise_convs com ZEROS para que,
        no step 0, o modelo se comporte exatamente como o BigVGAN V2 original.
        """
        print(f"\n{'='*20} LOADING NVIDIA BIGVGAN V2 PRE-TRAIN {'='*20}")
        print(f"Loading weights from: {checkpoint_path}")
        
        # 1. Carregar o checkpoint da NVIDIA
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # O checkpoint da NVIDIA geralmente tem a chave 'generator' ou 'model_g'
        if 'generator' in checkpoint:
            state_dict = checkpoint['generator']
        elif 'model_g' in checkpoint:
            state_dict = checkpoint['model_g']
        else:
            state_dict = checkpoint
        
        # 2. Carregar os pesos com strict=False
        # Isso permite que os pesos do caminho principal sejam carregados,
        # enquanto os novos pesos do NSF são ignorados e mantêm inicialização aleatória.
        missing_keys, unexpected_keys = self.generator.load_state_dict(state_dict, strict=False)
        
        print(f"Successfully loaded core weights.")
        print(f"Missing keys (Expected to be NSF/Noise related): {len(missing_keys)} keys")
        if len(missing_keys) > 0:
            print(f"Missing keys sample: {missing_keys[:5]}")
        
        # 3. O TRUQUE DE OURO: Inicializar as noise_convs com ZEROS
        zero_init_count = 0
        for name, module in self.generator.named_modules():
            if 'noise_convs' in name:
                if hasattr(module, 'weight') and module.weight is not None:
                    torch.nn.init.zeros_(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
                zero_init_count += 1
        
        print(f"Initialized {zero_init_count} NSF noise_convs layers to ZERO.")
        print(f"{'='*70}\n")

    def build_losses_and_metrics(self):
        """Instancia a classe de perda do BigVGAN V2."""
        self.mix_loss = BigVGANv2Loss(self.config)

    def Gforward(self, sample):
        """
        Forward pass do Gerador.
        Recebe o Mel e o F0, retorna o áudio gerado.
        """
        wav = self.generator(x=sample['mel'], f0=sample['f0'])
        return {'audio': wav}

    def G2forward(self, sample, pc_aug_num):
        """
        Forward pass com PC-Augmentation (Pitch Correction).
        Dobra o batch: metade com F0 original, metade com F0 shiftado.
        Isso força o modelo a aprender a corrigir o pitch.
        """
        if pc_aug_num <= 0:
            raise ValueError('pc_aug_num should be greater than 0')
        
        f0 = sample['f0']
        
        # Gera um shift aleatório de tom (em semitons)
        key_c = (2 * torch.rand(pc_aug_num, device=f0.device).unsqueeze(-1) - 1) * self.pc_aug_key
        
        # Shifta o F0 (limitando ao max_f0)
        f0_shift_c = torch.clip(f0[:pc_aug_num] * 2 ** (key_c / 12), max=self.max_f0)
        
        # Gera o áudio com o F0 shiftado
        wav_mixed = self.generator(
            x=sample['mel'],
            f0=torch.cat((f0_shift_c, f0[pc_aug_num:]), dim=0)
        )
        wav_shift_c, wav_shift_0 = wav_mixed[:pc_aug_num], wav_mixed[pc_aug_num:]
        
        # Recalcula o Mel do áudio shiftado
        mel_shift_c = self.TF.dynamic_range_compression_torch(self.TF(wav_shift_c.squeeze(1)))
        
        # Gera o áudio de volta com o F0 original e o Mel shiftado
        wav_shift_back = self.generator(x=mel_shift_c, f0=f0[:pc_aug_num])
        wav_ret = torch.cat((wav_shift_back, wav_shift_0), dim=0)
        
        return {
            'audio': wav_ret,
            'audio_shift_c': wav_shift_c,
        }

    def Dforward(self, Goutput):
        """
        Forward pass do Discriminator.
        O Discriminator do repo original retorna uma lista concatenada:
        r (MRD) + p (MPD) + s (MSD)
        Cada elemento é uma tupla (features, score).
        
        MRD: 3 discriminadores (3 resoluções) -> 3 tuplas
        MPD: 5 discriminadores (5 períodos) -> 5 tuplas
        MSD: 1 discriminador -> 1 tupla
        Total: 9 tuplas
        """
        disc_outputs = self.discriminator(Goutput)
        
        # Separar as saídas por discriminador
        # MRD: índices 0-2 (3 resoluções)
        mrd_out = []
        mrd_feature = []
        for i in range(3):
            features, score = disc_outputs[i]
            mrd_out.append(score)
            mrd_feature.append(features)
        
        # MPD: índices 3-7 (5 períodos)
        mpd_out = []
        mpd_feature = []
        for i in range(3, 8):
            features, score = disc_outputs[i]
            mpd_out.append(score)
            mpd_feature.append(features)
        
        # MSD: índice 8 (1 discriminador)
        msd_out = []
        msd_feature = []
        features, score = disc_outputs[8]
        msd_out.append(score)
        msd_feature.append(features)
        
        return {
            'mrd': (mrd_out, mrd_feature),
            'mpd': (mpd_out, mpd_feature),
            'msd': (msd_out, msd_feature),
        }

    def _training_step(self, sample, batch_idx):
        """
        Um passo de treinamento.
        Segue o padrão do GanBaseTask:
        1. Forward do Gerador
        2. Forward do Discriminator (com detach) e cálculo da loss do D
        3. Otimização do Discriminator
        4. Forward do Discriminator (sem detach) e cálculo da loss do G
        5. Cálculo da loss auxiliar (Mel + STFT)
        6. Otimização do Gerador
        """
        log_dict = {}
        opt_g, opt_d = self.optimizers()
        
        # === PC-AUGMENTATION (OPCIONAL) ===
        pc_aug_num = int(np.ceil(sample['audio'].shape[0] * self.pc_aug_rate))
        pc_aug = self.pc_aug and pc_aug_num > 0
        
        if pc_aug:
            Goutput = self.G2forward(sample=sample, pc_aug_num=pc_aug_num)
            audio_fake = torch.cat((
                Goutput['audio'],
                Goutput['audio_shift_c'],
            ), dim=0)
        else:
            Goutput = self.Gforward(sample=sample)
            audio_fake = Goutput['audio']
        
        # === TREINAMENTO DO DISCRIMINATOR ===
        Dfake = self.Dforward(Goutput=audio_fake.detach())
        Dtrue = self.Dforward(Goutput=sample['audio'])
        
        Dloss, Dlog = self.mix_loss.Dloss(Dfake=Dfake, Dtrue=Dtrue)
        log_dict.update(Dlog)
        
        opt_d.zero_grad()
        self.manual_backward(Dloss)
        if self.clip_grad_norm is not None:
            self.clip_gradients(opt_d, gradient_clip_val=self.clip_grad_norm, gradient_clip_algorithm="norm")
        opt_d.step()
        
        # Desabilita grad do Discriminator para o passo do Gerador
        d_trainable_params = [p for p in self.discriminator.parameters() if p.requires_grad]
        for p in d_trainable_params:
            p.requires_grad = False
        
        # === TREINAMENTO DO GERADOR ===
        GDfake = self.Dforward(Goutput=audio_fake)
        GDtrue = self.Dforward(Goutput=sample['audio'])
        
        GDloss, GDlog = self.mix_loss.GDloss(GDfake=GDfake, GDtrue=GDtrue)
        log_dict.update(GDlog)
        
        # Loss auxiliar (Mel + STFT)
        if pc_aug:
            # Adiciona a loss de PC-Wav (força o modelo a corrigir o pitch)
            pc_wav_loss = F.l1_loss(Goutput['audio_shift_c'], sample['audio'][:pc_aug_num]) * 30
            sample = {'audio': torch.cat((sample['audio'], sample['audio'][:pc_aug_num]), dim=0)}
            Goutput = {'audio': torch.cat((Goutput['audio'], Goutput['audio_shift_c']), dim=0)}
            log_dict['pc_wav_loss'] = pc_wav_loss.item()
        else:
            pc_wav_loss = 0
        
        Auxloss, Auxlog = self.mix_loss.Auxloss(Goutput=Goutput, sample=sample)
        Auxloss_total = Auxloss + pc_wav_loss
        log_dict.update(Auxlog)
        
        # Loss total do Gerador
        Gloss = GDloss + Auxloss_total
        
        opt_g.zero_grad()
        self.manual_backward(Gloss)
        if self.clip_grad_norm is not None:
            self.clip_gradients(opt_g, gradient_clip_val=self.clip_grad_norm, gradient_clip_algorithm="norm")
        opt_g.step()
        
        # Reabilita grad do Discriminator
        for p in d_trainable_params:
            p.requires_grad = True
        
        return log_dict

    def _validation_step(self, sample, batch_idx):
        """
        Um passo de validação.
        Gera o áudio, calcula a loss de STFT e loga no TensorBoard.
        """
        wav = self.Gforward(sample)['audio']
        
        with torch.no_grad():
            # Calcula o Mel do áudio gerado e do GT
            mel_fake = self.TF.dynamic_range_compression_torch(self.TF(wav.squeeze(1)))
            mel_real = self.TF.dynamic_range_compression_torch(self.TF(sample['audio'].squeeze(1)))
            
            if self.global_rank == 0:
                # Plota o Mel (real, fake e erro)
                self.plot_mel(batch_idx, mel_real, mel_fake, name=f'mel_{batch_idx}')
                
                # Loga o áudio gerado
                self.logger.experiment.add_audio(
                    f'BIGVGAN_{batch_idx}_', wav,
                    sample_rate=self.config['audio_sample_rate'],
                    global_step=self.global_step
                )
                
                # Loga o áudio GT apenas uma vez
                if batch_idx not in self.logged_gt_wav:
                    self.logger.experiment.add_audio(
                        f'gt_{batch_idx}_', sample['audio'],
                        sample_rate=self.config['audio_sample_rate'],
                        global_step=self.global_step
                    )
                    self.logged_gt_wav.add(batch_idx)
        
        # Calcula a loss de validação (L1 entre Mels)
        val_loss = nn.L1Loss()(mel_fake, mel_real)
        return {'val_mel_loss': val_loss}, 1

    def plot_mel(self, batch_idx, spec_real, spec_fake, name=None):
        """Plota o Mel (real, fake e erro) no TensorBoard."""
        name = f'mel_{batch_idx}' if name is None else name
        vmin = self.config.get('mel_vmin', -6.0)
        vmax = self.config.get('mel_vmax', 1.5)
        
        # Concatena: [erro + vmin, real, fake]
        spec_cat = torch.cat([(spec_fake - spec_real).abs() + vmin, spec_real, spec_fake], -1)
        self.logger.experiment.add_figure(
            name,
            spec_to_figure(spec_cat[0], vmin, vmax),
            self.global_step
        )
