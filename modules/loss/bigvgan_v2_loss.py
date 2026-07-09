import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.loss.stft_loss import warp_stft
from utils.wav2mel import PitchAdjustableMelSpectrogram

class BigVGANv2Loss(nn.Module):
    """
    Classe de Perda para o NSF-BigVGAN V2 integrado ao DiffSinger.
    Combina as perdas adversariais (LSGAN) e Feature Matching do BigVGAN
    com as perdas auxiliares (Mel + Multi-Resolution STFT) do ecossistema DiffSinger.
    """
    def __init__(self, config: dict):
        super().__init__()
        
        # ==========================================
        # 1. Extrator de Mel (Padrão DiffSinger)
        # ==========================================
        # Usa os parâmetros do YAML (44.1kHz, 128 bins, hop 512, fmin 40, fmax 16000)
        self.mel = PitchAdjustableMelSpectrogram(
            sample_rate=config['audio_sample_rate'],
            n_fft=config['fft_size'],
            win_length=config['win_size'],
            hop_length=config['hop_size'],
            f_min=config['fmin'],
            f_max=config['fmax_for_loss'], # Usa fmax_for_loss se existir, senão fmax
            n_mels=config['audio_num_mel_bins'],
        )
        
        self.L1loss = nn.L1Loss()
        
        # Pesos das perdas auxiliares (vindos do YAML)
        self.lab_aux_mel_loss = config.get('lab_aux_melloss', 45.0)
        self.lab_aux_stft_loss = config.get('lab_aux_stftloss', 2.5)
        
        # ==========================================
        # 2. Extrator de STFT (Multi-Resolution)
        # ==========================================
        # Usa as resoluções definidas no YAML para calcular a perda espectral
        if config.get('use_stftloss', True):
            self.stft = warp_stft({
                'fft_sizes': config['loss_fft_sizes'], 
                'hop_sizes': config['loss_hop_sizes'],
                'win_lengths': config['loss_win_lengths']
            })
        self.use_stftloss = config.get('use_stftloss', True)

    # ==========================================
    # 3. Perda do Discriminador (Dloss)
    # ==========================================
    def discriminator_loss(self, disc_real_outputs, disc_generated_outputs):
        """
        Calcula a perda LSGAN (Least Squares GAN) para o Discriminador.
        O Discriminador do BigVGAN retorna uma lista de tuplas: [(fmap, score), ...]
        """
        loss = 0
        rlosses = 0
        glosses = 0
        for (fmap_fake, score_fake), (fmap_real, score_real) in zip(disc_generated_outputs, disc_real_outputs):
            # Penaliza o discriminador se ele não der 1 para o áudio real
            r_loss = torch.mean((1 - score_real) ** 2)
            # Penaliza o discriminador se ele não der 0 para o áudio fake
            g_loss = torch.mean(score_fake ** 2)
            
            loss += r_loss + g_loss
            rlosses += r_loss.item()
            glosses += g_loss.item()
            
        return loss, rlosses, glosses

    def Dloss(self, Dfake, Dtrue):
        """
        Interface chamada pelo PyTorch Lightning (GanBaseTask).
        Dfake e Dtrue são as saídas do self.discriminator() (listas de tuplas).
        """
        loss, rlosses, glosses = self.discriminator_loss(Dtrue, Dfake)
        return loss, {
            'D_loss_real': rlosses, 
            'D_loss_fake': glosses
        }

    # ==========================================
    # 4. Perda do Gerador (GDloss)
    # ==========================================
    def feature_loss(self, fmap_r, fmap_g):
        """
        Calcula o Feature Matching Loss (L1 entre as features intermediárias).
        """
        loss = 0
        for dr, dg in zip(fmap_r, fmap_g):
            for rl, gl in zip(dr, dg):
                loss += torch.mean(torch.abs(rl - gl))
        # O paper do HiFi-GAN/BigVGAN multiplica o feature loss por 2
        return loss * 2

    def GDloss(self, GDfake, GDtrue):
        """
        Interface chamada pelo PyTorch Lightning.
        Calcula a perda adversarial do gerador + Feature Matching.
        """
        loss_gen = 0
        loss_feat = 0
        gen_losses_log = 0
        
        for (fmap_fake, score_fake), (fmap_real, score_real) in zip(GDfake, GDtrue):
            # 1. Perda Adversarial (Gerador tentando enganar o discriminador)
            l_gen = torch.mean((1 - score_fake) ** 2)
            loss_gen += l_gen
            gen_losses_log += l_gen.item()
            
            # 2. Feature Matching Loss
            loss_feat += self.feature_loss(fmap_real, fmap_fake)
            
        total_loss = loss_gen + loss_feat
        return total_loss, {
            'G_loss_adv': gen_losses_log, 
            'G_loss_feat': loss_feat.item()
        }

    # ==========================================
    # 5. Perdas Auxiliares (Auxloss)
    # ==========================================
    def Auxloss(self, Goutput, sample):
        """
        Calcula o Mel Loss (L1) e o Multi-Resolution STFT Loss.
        """
        wav_fake = Goutput['audio'].squeeze(1)
        wav_real = sample['audio'].squeeze(1)
        
        # Garante que os tamanhos batam (caso haja algum padding/corte mínimo)
        b = min(wav_fake.shape[0], wav_real.shape[0])
        wav_fake = wav_fake[:b]
        wav_real = wav_real[:b]
        
        # --- Mel Loss ---
        # Recalcula o Mel a partir do áudio gerado e real para garantir consistência
        mel_fake = self.mel.dynamic_range_compression_torch(self.mel(wav_fake))
        mel_real = self.mel.dynamic_range_compression_torch(self.mel(wav_real))
        mel_loss = self.L1loss(mel_fake, mel_real) * self.lab_aux_mel_loss
        
        # --- STFT Loss ---
        if self.use_stftloss:
            sc_loss, mag_loss = self.stft.stft(wav_fake, wav_real)
            stft_loss = (sc_loss + mag_loss) * self.lab_aux_stft_loss
            total_aux = mel_loss + stft_loss
            
            return total_aux, {
                'aux_mel_loss': mel_loss.item(), 
                'aux_stft_loss': stft_loss.item(),
                'aux_stft_sc': sc_loss.item(),
                'aux_stft_mag': mag_loss.item()
            }
        
        # Se não estiver usando STFT loss, retorna apenas o Mel Loss
        return mel_loss, {'aux_mel_loss': mel_loss.item()}
