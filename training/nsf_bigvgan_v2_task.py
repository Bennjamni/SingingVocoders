import pathlib
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
import torchaudio
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import Dataset
from models.nsf_bigvgan_v2.bigvgan_nsf import BigVGAN_NSF
from models.nsf_bigvgan_v2.env import AttrDict
from models.nsf_bigvgan_v2.discriminator import Discriminator
from modules.loss.bigvgan_v2_loss import BigVGANv2Loss
from training.base_task_gan import GanBaseTask
from utils.wav2F0 import PITCH_EXTRACTORS_ID_TO_NAME, get_pitch
from utils.wav2mel import PitchAdjustableMelSpectrogram

def spec_to_figure(spec, vmin=None, vmax=None):
    if isinstance(spec, torch.Tensor):
        spec = spec.cpu().numpy()
    fig = plt.figure(figsize=(12, 9), dpi=100)
    plt.pcolor(spec.T, vmin=vmin, vmax=vmax)
    plt.tight_layout()
    return fig

def dynamic_range_compression_torch(x, C=1, clip_val=1e-9):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def wav_aug(wav, hop_size, speed=1):
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
    source_sr = config['audio_sample_rate']
    max_f0 = source_sr / 2
    return max_f0

class nsf_bigvgan_v2_dataset(Dataset):
    def __init__(self, config: dict, data_dir, infer=False):
        super().__init__()
        self.config = config
        self.data_dir = data_dir if isinstance(data_dir, pathlib.Path) else pathlib.Path(data_dir)
        with open(self.data_dir, 'r', encoding='utf8') as f:
            fills = f.read().strip().split('\n')
        self.data_index = fills
        self.infer = infer
        self.volume_aug = self.config['volume_aug']
        self.volume_aug_prob = self.config['volume_aug_prob'] if not infer else 0
        self.key_aug = self.config.get('key_aug', False)
        self.key_aug_prob = self.config.get('key_aug_prob', 0.5)
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
        if sample['f0'].max() >= self.max_f0:
            return self.__getitem__(random.randint(0, len(self) - 1))
        return sample

    def __len__(self):
        return len(self.data_index)

    def get_data(self, index):
        data_path = pathlib.Path(self.data_index[index])
        data = np.load(data_path)
        pe_name = PITCH_EXTRACTORS_ID_TO_NAME[int(data['pe'])]
        if self.infer or not self.key_aug or random.random() > self.key_aug_prob:
            return {'f0': data['f0'], 'spectrogram': data['mel'], 'audio': data['audio']}
        else:
            speed = random.uniform(self.config['aug_min'], self.config['aug_max'])
            crop_mel_frames = int(np.ceil((self.config['crop_mel_frames'] + 4) * speed))
            samples_per_frame = self.config['hop_size']
            crop_wav_samples = crop_mel_frames * samples_per_frame
            if crop_wav_samples >= data['audio'].shape[0]:
                return {'f0': data['f0'], 'spectrogram': data['mel'], 'audio': data['audio']}
            start = random.randint(0, data['audio'].shape[0] - 1 - crop_wav_samples)
            end = start + crop_wav_samples
            audio = data['audio'][start:end]
            audio_aug = wav_aug(torch.from_numpy(audio), self.config["hop_size"], speed=speed)
            mel_aug = dynamic_range_compression_torch(self.mel_spec_transform(audio_aug[None, :]))
            f0, uv = get_pitch(
                pe_name, audio, length=mel_aug.shape[-1], hparams=self.config,
                speed=speed, interp_uv=True
            )
            if f0 is None:
                return {'f0': data['f0'], 'spectrogram': data['mel'], 'audio': data['audio']}
            audio_aug = audio_aug[2 * samples_per_frame: -2 * samples_per_frame].numpy()
            mel_aug = mel_aug[0, :, 2:-2].T.numpy()
            f0_aug = f0[2:-2] * speed
            return {'f0': f0_aug, 'spectrogram': mel_aug, 'audio': audio_aug}

    def collater(self, minibatch):
        samples_per_frame = self.config['hop_size']
        if self.infer:
            crop_mel_frames = 0
        else:
            crop_mel_frames = self.config['crop_mel_frames']
        for record in minibatch:
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
                record['audio'] = np.pad(record['audio'], (
                    0, (len(record['spectrogram'].T) * samples_per_frame) - len(record['audio'])),
                                         mode='constant')
            else:
                record['audio'] = record['audio'][start:end]
                record['audio'] = np.pad(record['audio'], (0, (end - start) - len(record['audio'])),
                                         mode='constant')
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
        audio = np.stack([record['audio'] for record in minibatch if 'audio' in record])
        spectrogram = np.stack([record['spectrogram'] for record in minibatch if 'spectrogram' in record])
        f0 = np.stack([record['f0'] for record in minibatch if 'f0' in record])
        return {
            'audio': torch.from_numpy(audio).unsqueeze(1),
            'mel': torch.from_numpy(spectrogram), 'f0': torch.from_numpy(f0),
        }

class stftlog:
    def __init__(self,
                 n_fft=2048,
                 win_length=2048,
                 hop_length=512,
                 center=False, ):
        self.hop_length = hop_length
        self.win_size = win_length
        self.n_fft = n_fft
        self.win_size = win_length
        self.center = center
        self.hann_window = {}

    def exc(self, y):
        hann_window_key = f"{y.device}"
        if hann_window_key not in self.hann_window:
            self.hann_window[hann_window_key] = torch.hann_window(
                self.win_size, device=y.device
            )
        y = torch.nn.functional.pad(
            y.unsqueeze(1),
            (
                int((self.win_size - self.hop_length) // 2),
                int((self.win_size - self.hop_length + 1) // 2),
            ),
            mode="reflect",
        )
        y = y.squeeze(1)
        spec = torch.stft(
            y,
            self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_size,
            window=self.hann_window[hann_window_key],
            center=self.center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        ).abs()
        return spec

class nsf_bigvgan_v2_task(GanBaseTask):
    def __init__(self, config):
        super().__init__(config)
        self.TF = PitchAdjustableMelSpectrogram(
            sample_rate=config['audio_sample_rate'],
            n_fft=config['fft_size'],
            win_length=config['win_size'],
            hop_length=config['hop_size'],
            f_min=config['fmin'],
            f_max=config['fmax'],
            n_mels=config['audio_num_mel_bins'],
        )
        self.pc_aug = self.config.get('pc_aug', False)
        self.pc_aug_rate = self.config.get('pc_aug_rate', 0.5)
        self.pc_aug_key = self.config.get('pc_aug_key', 5)
        self.logged_gt_wav = set()
        self.stft = stftlog()
        self.max_f0 = get_max_f0_from_config(config)

    def build_dataset(self):
        self.train_dataset = nsf_bigvgan_v2_dataset(config=self.config,
                                              data_dir=pathlib.Path(self.config['DataIndexPath']) / self.config[
                                                  'train_set_name'])
        self.valid_dataset = nsf_bigvgan_v2_dataset(config=self.config,
                                              data_dir=pathlib.Path(self.config['DataIndexPath']) / self.config[
                                                  'valid_set_name'], infer=True)

    def build_model(self):
        cfg = self.config['model_args']
        cfg.update({
            'sampling_rate': self.config['audio_sample_rate'],
            'num_mels': self.config['audio_num_mel_bins'],
            'hop_size': self.config['hop_size']
        })
        h = AttrDict(cfg)
        
        # Instancia o Gerador Híbrido (BigVGAN V2 + NSF)
        self.generator = BigVGAN_NSF(h, use_cuda_kernel=False)
        
        # Instancia os Discriminadores (MPD + MRD + MSD)
        self.discriminator = nn.ModuleDict({
            'mpd': self.generator.m_source.l_sin_gen,  # Placeholder, será substituído
            'mrd': nn.ModuleDict(),  # Placeholder
            'msd': nn.ModuleDict()   # Placeholder
        })
        
        # Na verdade, o Discriminator do BigVGAN V2 já empacota MPD + MRD + MSD
        # Vamos usar a classe Discriminator do BigVGAN V2 diretamente
        self.discriminator = Discriminator(h)

    def build_losses_and_metrics(self):
        self.mix_loss = BigVGANv2Loss(self.config)

    def Gforward(self, sample):
        """
        steps:
            1. run the full model
            2. calculate losses if not infer
        """
        wav = self.generator(x=sample['mel'], f0=sample['f0'])
        return {'audio': wav}

    def Dforward(self, Goutput):
        # O Discriminator do BigVGAN V2 retorna uma lista de tuplas (features, score)
        # Precisamos adaptar para o formato esperado pelo BigVGANv2Loss
        disc_outputs = self.discriminator(Goutput)
        
        # Separar MPD, MRD e MSD
        mpd_out = []
        mrd_out = []
        msd_out = []
        mpd_feature = []
        mrd_feature = []
        msd_feature = []
        
        for i, (features, score) in enumerate(disc_outputs):
            if i < 5:  # MPD (5 períodos)
                mpd_out.append(score)
                mpd_feature.append(features)
            elif i < 8:  # MRD (3 resoluções)
                mrd_out.append(score)
                mrd_feature.append(features)
            else:  # MSD (1 discriminador)
                msd_out.append(score)
                msd_feature.append(features)
        
        return {
            'mpd': (mpd_out, mpd_feature),
            'mrd': (mrd_out, mrd_feature),
            'msd': (msd_out, msd_feature)
        }

    def _training_step(self, sample, batch_idx):
        """
        :return: total loss: torch.Tensor, loss_log: dict, other_log: dict
        """
        log_dict = {}
        opt_g, opt_d = self.optimizers()

        # Forward Generator
        Goutput = self.Gforward(sample=sample)
        audio_fake = Goutput['audio']

        # Forward Discriminator (com detach)
        Dfake = self.Dforward(Goutput=audio_fake.detach())
        Dtrue = self.Dforward(Goutput=sample['audio'])

        # Loss do Discriminator
        Dloss, Dlog = self.mix_loss.Dloss(Dfake=Dfake, Dtrue=Dtrue)
        log_dict.update(Dlog)

        # Otimizar Discriminator
        opt_d.zero_grad()
        self.manual_backward(Dloss)
        if self.clip_grad_norm is not None:
            self.clip_gradients(opt_d, gradient_clip_val=self.clip_grad_norm, gradient_clip_algorithm="norm")
        opt_d.step()

        # Desabilitar grad do Discriminator
        d_trainable_params = [p for p in self.discriminator.parameters() if p.requires_grad]
        for p in d_trainable_params:
            p.requires_grad = False

        # Forward Discriminator (sem detach)
        GDfake = self.Dforward(Goutput=audio_fake)
        GDtrue = self.Dforward(Goutput=sample['audio'])

        # Loss do Generator (Feature Matching + Score Loss)
        GDloss, GDlog = self.mix_loss.GDloss(GDfake=GDfake, GDtrue=GDtrue)
        log_dict.update(GDlog)

        # Loss Auxiliar (Mel Loss + STFT Loss)
        Auxloss, Auxlog = self.mix_loss.Auxloss(Goutput=Goutput, sample=sample)
        log_dict.update(Auxlog)

        # Loss total do Generator
        Gloss = GDloss + Auxloss

        # Otimizar Generator
        opt_g.zero_grad()
        self.manual_backward(Gloss)
        if self.clip_grad_norm is not None:
            self.clip_gradients(opt_g, gradient_clip_val=self.clip_grad_norm, gradient_clip_algorithm="norm")
        opt_g.step()

        # Reabilitar grad do Discriminator
        for p in d_trainable_params:
            p.requires_grad = True

        return log_dict

    def _validation_step(self, sample, batch_idx):
        wav = self.Gforward(sample)['audio']
        with torch.no_grad():
            stfts = self.stft.exc(wav.squeeze(0).cpu().float())
            Gstfts = self.stft.exc(sample['audio'].squeeze(0).cpu().float())
            stfts_log10 = torch.log10(torch.clamp(stfts, min=1e-7))
            Gstfts_log10 = torch.log10(torch.clamp(Gstfts, min=1e-7))
            if self.global_rank == 0:
                self.plot_mel(batch_idx, Gstfts_log10.transpose(1, 2), stfts_log10.transpose(1, 2),
                              name=f'log10stft_{batch_idx}')
                self.logger.experiment.add_audio(f'BIGVGAN_{batch_idx}_', wav,
                                                 sample_rate=self.config['audio_sample_rate'],
                                                 global_step=self.global_step)
                if batch_idx not in self.logged_gt_wav:
                    self.logger.experiment.add_audio(f'gt_{batch_idx}_', sample['audio'],
                                                     sample_rate=self.config['audio_sample_rate'],
                                                     global_step=self.global_step)
                    self.logged_gt_wav.add(batch_idx)
        return {'stft_loss': nn.L1Loss()(Gstfts_log10, stfts_log10)}, 1

    def plot_mel(self, batch_idx, spec, spec_out, name=None):
        name = f'mel_{batch_idx}' if name is None else name
        vmin = self.config['mel_vmin']
        vmax = self.config['mel_vmax']
        spec_cat = torch.cat([(spec_out - spec).abs() + vmin, spec, spec_out], -1)
        self.logger.experiment.add_figure(name, spec_to_figure(spec_cat[0], vmin, vmax), self.global_step)
