"""Bounded held-out reconstruction comparison; not a perceptual quality metric."""
from pathlib import Path


def evaluate(model, loader, folder: Path, device, sample_rate, seed, stopped=lambda: False):
    import numpy as np
    import soundfile as sf
    import torch
    scores, files, lengths = [], [], []
    was_training = model.training
    model.eval()
    try:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()] if device.startswith("cuda") else []), torch.inference_mode():
            torch.manual_seed(seed)
            for i, info in enumerate(loader):
                if i >= 8 or i and stopped(): break
                phone, phone_lengths, pitch, pitchf, _, _, wave, wave_lengths, sid = info
                generated, *_ = model.infer(*[x.to(device) for x in (phone, phone_lengths, pitch, pitchf, sid)])
                target = wave[0, 0, :int(wave_lengths[0])].to(device)
                output = generated[0, 0]
                length = min(len(target), len(output))
                if length < 1024: raise ValueError("RVC validation produced too little audio")
                win = torch.hann_window(1024, device=device)
                def spectrum(x):
                    return torch.log(torch.stft(x[:length].float(), n_fft=1024, hop_length=256, window=win, return_complex=True).abs().clamp_min(1e-5))
                score = float((spectrum(target) - spectrum(output)).abs().mean())
                if not np.isfinite(score): raise ValueError("Non-finite held-out RVC spectral error")
                scores.append(score); lengths.append(abs(len(target)-len(output)))
                if i == 0:
                    for name, audio in (("heldout-original.wav", target), ("heldout-generated.wav", output)):
                        sf.write(folder / name, audio.float().cpu().numpy(), sample_rate)
                        files.append(str(folder / name))
    finally: model.train(was_training)
    if not scores: raise ValueError("No usable held-out RVC chunks")
    return dict(heldout_log_spectral_l1=sum(scores)/len(scores), chunks=len(scores), mean_length_error_samples=sum(lengths)/len(lengths)), files
