#!/usr/bin/env python3
"""
3-Fold Cross-Validation for 2×2 Factorial Benchmark.

Option A: Train on 2 patients, test on 1 (3 folds)

Fold structure:
  Fold 1: Train P1+P2 (80%), Val P1+P2 (20%), Test P5
  Fold 2: Train P1+P5 (80%), Val P1+P5 (20%), Test P2
  Fold 3: Train P2+P5 (80%), Val P2+P5 (20%), Test P1

This provides 3 independent test results for robust statistical analysis.

USAGE:
    # Run single fold
    python train_2um_cv_3fold.py --decoder hist2st --loss poisson --fold 1

    # Run all folds
    python train_2um_cv_3fold.py --decoder hist2st --loss poisson --fold all
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim_metric
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision import transforms
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

torch.set_float32_matmul_precision('high')

# CV fold definitions
CV_FOLDS = {
    1: {'train': ['P1', 'P2'], 'test': 'P5'},
    2: {'train': ['P1', 'P5'], 'test': 'P2'},
    3: {'train': ['P2', 'P5'], 'test': 'P1'},
}


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class JointGeometricTransform:
    def __init__(self, p_hflip=0.5, p_vflip=0.5, p_rot90=0.5):
        self.p_hflip = p_hflip
        self.p_vflip = p_vflip
        self.p_rot90 = p_rot90

    def __call__(self, image, label_2um, label_8um, mask_2um):
        if torch.rand(1).item() < self.p_hflip:
            image = torch.flip(image, dims=[2])
            label_2um = torch.flip(label_2um, dims=[2])
            label_8um = torch.flip(label_8um, dims=[2])
            mask_2um = torch.flip(mask_2um, dims=[2])
        if torch.rand(1).item() < self.p_vflip:
            image = torch.flip(image, dims=[1])
            label_2um = torch.flip(label_2um, dims=[1])
            label_8um = torch.flip(label_8um, dims=[1])
            mask_2um = torch.flip(mask_2um, dims=[1])
        if torch.rand(1).item() < self.p_rot90:
            k = 1
            image = torch.rot90(image, k, dims=[1, 2])
            label_2um = torch.rot90(label_2um, k, dims=[1, 2])
            label_8um = torch.rot90(label_8um, k, dims=[1, 2])
            mask_2um = torch.rot90(mask_2um, k, dims=[1, 2])
        return image, label_2um, label_8um, mask_2um


model_base = Path(os.environ.get('VISIUM_MODEL_BASE', '/home/user/visium-hd-2um-benchmark'))
sys.path.insert(0, str(model_base))

from model.encoder_wrapper import get_spatial_encoder
from model.enhanced_decoder import Hist2STDecoder


def log_epoch(log_file, epoch_data):
    def convert_to_native(obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return [convert_to_native(x) for x in obj.tolist()]
        elif isinstance(obj, list):
            return [convert_to_native(x) for x in obj]
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        return obj
    epoch_data = convert_to_native(epoch_data)
    with open(log_file, 'a') as f:
        f.write(json.dumps(epoch_data) + '\n')


def get_git_commit():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()
    except:
        return 'unknown'


def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch) / float(max(1, warmup_epochs))
        progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class RawCountsSTDataset(Dataset):
    def __init__(self, data_dir, patient_id, num_genes=50, transform=None,
                 input_size=224, joint_transform=None):
        self.data_dir = Path(data_dir) / patient_id
        self.patient_id = patient_id
        self.num_genes = num_genes
        self.transform = transform
        self.joint_transform = joint_transform
        self.input_size = input_size

        patches_file = self.data_dir / 'patches_raw_counts.npy'
        self.patches = np.load(patches_file, allow_pickle=True).tolist()

        with open(self.data_dir / 'gene_names.json') as f:
            self.gene_names = json.load(f)
            if isinstance(self.gene_names, dict):
                self.gene_names = self.gene_names.get('gene_names', list(self.gene_names.keys()))

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        item = self.patches[idx]
        img_path_key = 'img_path' if 'img_path' in item else 'image_path'
        img_name = Path(item[img_path_key]).name
        img_path = self.data_dir / 'images' / img_name

        if not img_path.exists():
            alt = Path('/mnt/x/img2st_rotation_demo/processed_crc_raw_counts') / self.patient_id / 'images' / img_name
            if alt.exists():
                img_path = alt

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        counts_2um_key = 'label_2um' if 'label_2um' in item else 'counts_2um'
        label_2um_flat = torch.tensor(item[counts_2um_key], dtype=torch.float32)
        label_2um = label_2um_flat.reshape(128, 128, -1)[:, :, :self.num_genes].permute(2, 0, 1)
        mask_2um = torch.tensor(item['mask_2um'], dtype=torch.float32).unsqueeze(0)

        label_4um = F.avg_pool2d(label_2um.unsqueeze(0), 2, 2).squeeze(0) * 4
        label_8um = F.avg_pool2d(label_4um.unsqueeze(0), 2, 2).squeeze(0) * 4

        if self.joint_transform is not None:
            image, label_2um, label_8um, mask_2um = self.joint_transform(image, label_2um, label_8um, mask_2um)

        return {
            'image': image,
            'label_2um': label_2um,
            'label_8um': label_8um,
            'mask_2um': mask_2um,
        }


class MiniUNet(nn.Module):
    """Simple UNet-style decoder (Img2ST architecture)"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(in_channels, 512, 2, stride=2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU()
        )
        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU()
        )
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU()
        )
        self.conv4 = nn.Conv2d(128, out_channels, 1)

    def forward(self, x):
        x = self.up1(x)
        x = self.conv1(x)
        x = self.up2(x)
        x = self.conv2(x)
        x = self.up3(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x


class DecoderModel(nn.Module):
    """Unified model supporting both decoder types and loss types"""

    def __init__(self, encoder_name, decoder_type, loss_type, num_genes=50, input_size=224):
        super().__init__()
        self.decoder_type = decoder_type
        self.loss_type = loss_type

        self.encoder = get_spatial_encoder(encoder_name)
        for param in self.encoder.parameters():
            param.requires_grad = False

        enc_dim = 1024

        if decoder_type == 'hist2st':
            self.decoder = Hist2STDecoder(
                in_ch=enc_dim, hidden_ch=512, out_ch=512,
                num_heads=8, k_neighbors=8, dropout=0.1
            )
            self.head = nn.Conv2d(512, num_genes, 1)
            if loss_type == 'poisson':
                # Initialize bias to -3.0 so exp(-3.0) ≈ 0.05, matching sparse data prior
                # where ~95% of 2µm bins have zero UMI counts
                nn.init.constant_(self.head.bias, -3.0)
        elif decoder_type == 'img2st':
            self.decoder = MiniUNet(enc_dim, num_genes)
            self.head = None
            if loss_type == 'poisson':
                nn.init.constant_(self.decoder.conv4.bias, -3.0)
        else:
            raise ValueError(f"Unknown decoder: {decoder_type}")

        self.upsample = nn.Upsample(size=(128, 128), mode='bilinear', align_corners=False)
        self.softplus = nn.Softplus() if loss_type == 'mse' else None

    def forward(self, images):
        with torch.no_grad():
            features = self.encoder(images)

        pred = self.decoder(features)

        if self.head is not None:
            pred = self.head(pred)

        if pred.shape[-1] != 128:
            pred = self.upsample(pred)

        if self.softplus is not None:
            pred = self.softplus(pred)

        return pred


def train_epoch_poisson(model, loader, optimizer, device, grad_accum=1, scaler=None):
    """Training with Poisson NLL loss"""
    model.train()
    total_loss = 0
    n_batches = 0
    optimizer.zero_grad()

    total_batches = len(loader)
    last_window_size = total_batches % grad_accum or grad_accum

    for batch_idx, batch in enumerate(tqdm(loader, desc='Training', leave=False)):
        images = batch['image'].to(device, non_blocking=True)
        label_2um = batch['label_2um'].to(device, non_blocking=True)
        mask_2um = batch['mask_2um'].to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            pred_log_rate = model(images)
            mask_expanded = mask_2um.expand_as(pred_log_rate)
            pred_log_rate = torch.clamp(pred_log_rate, max=20.0)
            rate = torch.exp(pred_log_rate)
            nll = rate - label_2um * pred_log_rate
            valid_mask = mask_expanded > 0.5
            n_valid = valid_mask.sum()

            if n_valid > 0:
                loss = (nll * valid_mask.float()).sum() / n_valid
            else:
                loss = torch.tensor(0.0, device=device, requires_grad=True)

        is_last_batch = (batch_idx + 1) == total_batches
        is_step_batch = (batch_idx + 1) % grad_accum == 0
        is_last_window = batch_idx >= (total_batches - last_window_size)
        actual_accum = last_window_size if is_last_window else grad_accum

        scaled_loss = loss / actual_accum
        if scaler is not None:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        if is_step_batch or is_last_batch:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def train_epoch_mse(model, loader, optimizer, device, grad_accum=1, scaler=None):
    """Training with MSE loss"""
    model.train()
    total_loss = 0
    n_batches = 0
    optimizer.zero_grad()

    total_batches = len(loader)
    last_window_size = total_batches % grad_accum or grad_accum

    for batch_idx, batch in enumerate(tqdm(loader, desc='Training', leave=False)):
        images = batch['image'].to(device, non_blocking=True)
        label_2um = batch['label_2um'].to(device, non_blocking=True)
        mask_2um = batch['mask_2um'].to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            pred = model(images)
            mask_expanded = mask_2um.expand_as(pred)
            valid_mask = mask_expanded > 0.5
            n_valid = valid_mask.sum()

            if n_valid > 0:
                mse = (pred - label_2um) ** 2
                loss = (mse * valid_mask.float()).sum() / n_valid
            else:
                loss = torch.tensor(0.0, device=device, requires_grad=True)

        is_last_batch = (batch_idx + 1) == total_batches
        is_step_batch = (batch_idx + 1) % grad_accum == 0
        is_last_window = batch_idx >= (total_batches - last_window_size)
        actual_accum = last_window_size if is_last_window else grad_accum

        scaled_loss = loss / actual_accum
        if scaler is not None:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        if is_step_batch or is_last_batch:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device, loss_type='poisson'):
    """Evaluate model on a dataset"""
    model.eval()
    all_pred, all_label, all_mask = [], [], []

    for batch in tqdm(loader, desc='Evaluating', leave=False):
        images = batch['image'].to(device)
        pred = model(images)

        if loss_type == 'poisson':
            pred = torch.exp(torch.clamp(pred, max=20.0))

        all_pred.append(pred.cpu().numpy())
        all_label.append(batch['label_2um'].numpy())
        all_mask.append(batch['mask_2um'].numpy())

    pred_2um = np.concatenate(all_pred)
    label_2um = np.concatenate(all_label)
    mask_2um = np.concatenate(all_mask)

    n_genes = pred_2um.shape[1]

    # Global PCC at 2um
    mask_flat = mask_2um[:, 0].flatten()
    valid_idx = mask_flat > 0.5
    pred_flat = pred_2um.transpose(0, 2, 3, 1).reshape(-1, n_genes)[valid_idx]
    label_flat = label_2um.transpose(0, 2, 3, 1).reshape(-1, n_genes)[valid_idx]

    gene_pccs = []
    for g in range(n_genes):
        p, l = pred_flat[:, g], label_flat[:, g]
        if p.std() > 1e-6 and l.std() > 1e-6:
            gene_pccs.append(pearsonr(p, l)[0])

    pcc_2um = np.mean(gene_pccs) if gene_pccs else 0.0

    # PCC at 8um
    pred_8um = pred_2um.reshape(pred_2um.shape[0], n_genes, 32, 4, 32, 4).mean(axis=(3, 5))
    label_8um = label_2um.reshape(label_2um.shape[0], n_genes, 32, 4, 32, 4).mean(axis=(3, 5))
    mask_8um = mask_2um[:, 0].reshape(mask_2um.shape[0], 32, 4, 32, 4).max(axis=(2, 4))

    mask_flat_8 = mask_8um.flatten()
    valid_idx_8 = mask_flat_8 > 0.5
    pred_flat_8 = pred_8um.transpose(0, 2, 3, 1).reshape(-1, n_genes)[valid_idx_8]
    label_flat_8 = label_8um.transpose(0, 2, 3, 1).reshape(-1, n_genes)[valid_idx_8]

    gene_pccs_8 = []
    for g in range(n_genes):
        p, l = pred_flat_8[:, g], label_flat_8[:, g]
        if p.std() > 1e-6 and l.std() > 1e-6:
            gene_pccs_8.append(pearsonr(p, l)[0])

    pcc_8um = np.mean(gene_pccs_8) if gene_pccs_8 else 0.0

    # SSIM at 2um
    ssim_2um_list = []
    for b in range(pred_2um.shape[0]):
        if mask_2um[b, 0].mean() > 0.05:
            for g in range(n_genes):
                p_img = pred_2um[b, g] * mask_2um[b, 0]
                l_img = label_2um[b, g] * mask_2um[b, 0]
                combined = np.concatenate([p_img.flatten(), l_img.flatten()])
                vmin, vmax = combined.min(), combined.max()
                if vmax - vmin > 1e-6:
                    try:
                        s = ssim_metric((p_img - vmin) / (vmax - vmin),
                                       (l_img - vmin) / (vmax - vmin), data_range=1.0)
                        if not np.isnan(s):
                            ssim_2um_list.append(s)
                    except:
                        pass
    ssim_2um = np.mean(ssim_2um_list) if ssim_2um_list else 0.0

    return {
        'pcc_2um': pcc_2um, 'pcc_8um': pcc_8um, 'ssim_2um': ssim_2um,
        'pred_2um': pred_2um, 'label_2um': label_2um, 'mask_2um': mask_2um,
    }


def train_fold(fold_num, args, device):
    """Train a single fold"""
    fold_config = CV_FOLDS[fold_num]
    train_patients = fold_config['train']
    test_patient = fold_config['test']

    print(f"\n{'='*70}")
    print(f"FOLD {fold_num}: Train {train_patients}, Test {test_patient}")
    print(f"{'='*70}")

    # Model naming
    model_names = {
        ('hist2st', 'mse'): "D'",
        ('hist2st', 'poisson'): "E'",
        ('img2st', 'poisson'): "F",
        ('img2st', 'mse'): "G",
    }
    model_name = model_names[(args.decoder, args.loss)]

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = Path(args.save_dir) / f'model_{model_name.replace("'", "_prime")}_{args.decoder}_{args.loss}' / f'fold{fold_num}_test{test_patient}_{timestamp}'
    save_dir.mkdir(parents=True, exist_ok=True)

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_joint_transform = JointGeometricTransform()

    eval_transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load training data (combined from 2 patients)
    train_datasets = [RawCountsSTDataset(args.data_dir, p, args.num_genes, train_transform,
                                          joint_transform=train_joint_transform)
                      for p in train_patients]
    full_train_dataset = ConcatDataset(train_datasets)

    # Split 80/20 for train/val
    n_total = len(full_train_dataset)
    n_val = int(n_total * args.val_fraction)
    n_train = n_total - n_val

    # Deterministic split based on seed
    generator = torch.Generator().manual_seed(args.seed + fold_num)
    indices = torch.randperm(n_total, generator=generator).tolist()
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(full_train_dataset, val_indices)

    # Test dataset
    test_dataset = RawCountsSTDataset(args.data_dir, test_patient, args.num_genes, eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Save config
    config = {
        'fold': fold_num,
        'train_patients': train_patients,
        'test_patient': test_patient,
        'val_fraction': args.val_fraction,
        'n_train': n_train,
        'n_val': n_val,
        'n_test': len(test_dataset),
        'decoder': args.decoder,
        'loss': args.loss,
        'model_name': model_name,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'seed': args.seed,
        'git_commit': get_git_commit(),
    }
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Model
    model = DecoderModel(args.encoder, args.decoder, args.loss, args.num_genes, args.input_size).to(device)

    if args.decoder == 'hist2st':
        trainable_params = list(model.decoder.parameters()) + list(model.head.parameters())
    else:
        trainable_params = list(model.decoder.parameters())

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_epochs, args.epochs)

    train_fn = train_epoch_poisson if args.loss == 'poisson' else train_epoch_mse

    best_val_ssim = -1
    patience_counter = 0
    log_file = save_dir / 'training_log.jsonl'

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        # Train
        train_loss = train_fn(model, train_loader, optimizer, device, args.grad_accum)
        scheduler.step()

        # Validate
        val_metrics = evaluate(model, val_loader, device, args.loss)
        val_ssim = val_metrics['ssim_2um']
        val_pcc = val_metrics['pcc_2um']

        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val SSIM: {val_ssim:.4f}, Val PCC: {val_pcc:.4f}")

        # Log
        log_epoch(log_file, {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_ssim_2um': val_ssim,
            'val_pcc_2um': val_pcc,
            'val_pcc_8um': val_metrics['pcc_8um'],
            'lr': optimizer.param_groups[0]['lr'],
        })

        # Early stopping on validation SSIM
        if val_ssim > best_val_ssim:
            best_val_ssim = val_ssim
            patience_counter = 0
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_ssim': best_val_ssim,
            }, save_dir / 'best_model.pt')
            print(f"  -> New best! Saved checkpoint")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    # Load best model for final test
    print(f"\nLoading best model (Val SSIM: {best_val_ssim:.4f})")
    checkpoint = torch.load(save_dir / 'best_model.pt', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Final test evaluation
    print(f"Evaluating on test set ({test_patient})...")
    test_metrics = evaluate(model, test_loader, device, args.loss)

    print(f"\n{'='*50}")
    print(f"FOLD {fold_num} FINAL RESULTS (Test: {test_patient})")
    print(f"{'='*50}")
    print(f"Test SSIM 2µm: {test_metrics['ssim_2um']:.4f}")
    print(f"Test PCC 2µm:  {test_metrics['pcc_2um']:.4f}")
    print(f"Test PCC 8µm:  {test_metrics['pcc_8um']:.4f}")

    # Save test metrics
    final_metrics = {
        'fold': fold_num,
        'test_patient': test_patient,
        'best_val_ssim': float(best_val_ssim),
        'test_ssim_2um': float(test_metrics['ssim_2um']),
        'test_pcc_2um': float(test_metrics['pcc_2um']),
        'test_pcc_8um': float(test_metrics['pcc_8um']),
    }
    with open(save_dir / 'final_metrics.json', 'w') as f:
        json.dump(final_metrics, f, indent=2)

    # Save predictions
    np.save(save_dir / 'pred_2um.npy', test_metrics['pred_2um'])
    np.save(save_dir / 'label_2um.npy', test_metrics['label_2um'])
    np.save(save_dir / 'mask_2um.npy', test_metrics['mask_2um'])

    return final_metrics


def main():
    parser = argparse.ArgumentParser(description='3-Fold Cross-Validation for 2×2 Factorial')

    # Model configuration
    parser.add_argument('--decoder', type=str, choices=['hist2st', 'img2st'], required=True)
    parser.add_argument('--loss', type=str, choices=['mse', 'poisson'], required=True)

    # Fold selection
    parser.add_argument('--fold', type=str, default='all',
                        help='Fold to run: 1, 2, 3, or "all"')

    # Data
    parser.add_argument('--data_dir', type=str,
                        default='/mnt/x/img2st_rotation_demo/processed_crc_raw_counts')
    parser.add_argument('--save_dir', type=str,
                        default='/mnt/x/mse-vs-poisson-2um-benchmark/results_cv')
    parser.add_argument('--val_fraction', type=float, default=0.2,
                        help='Fraction of training data for validation')

    # Model
    parser.add_argument('--encoder', type=str, default='virchow2')
    parser.add_argument('--num_genes', type=int, default=50)
    parser.add_argument('--input_size', type=int, default=224)

    # Training
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--grad_accum', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--warmup_epochs', type=int, default=2)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Determine which folds to run
    if args.fold.lower() == 'all':
        folds = [1, 2, 3]
    else:
        folds = [int(args.fold)]

    all_results = []
    for fold in folds:
        result = train_fold(fold, args, device)
        all_results.append(result)

    # Print summary
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print("CROSS-VALIDATION SUMMARY")
        print(f"{'='*70}")
        ssim_values = [r['test_ssim_2um'] for r in all_results]
        pcc_2um_values = [r['test_pcc_2um'] for r in all_results]
        pcc_8um_values = [r['test_pcc_8um'] for r in all_results]

        print(f"SSIM 2µm: {np.mean(ssim_values):.4f} ± {np.std(ssim_values):.4f}")
        print(f"PCC 2µm:  {np.mean(pcc_2um_values):.4f} ± {np.std(pcc_2um_values):.4f}")
        print(f"PCC 8µm:  {np.mean(pcc_8um_values):.4f} ± {np.std(pcc_8um_values):.4f}")

        # Save summary
        summary = {
            'decoder': args.decoder,
            'loss': args.loss,
            'folds': all_results,
            'mean_ssim_2um': float(np.mean(ssim_values)),
            'std_ssim_2um': float(np.std(ssim_values)),
            'mean_pcc_2um': float(np.mean(pcc_2um_values)),
            'std_pcc_2um': float(np.std(pcc_2um_values)),
            'mean_pcc_8um': float(np.mean(pcc_8um_values)),
            'std_pcc_8um': float(np.std(pcc_8um_values)),
        }
        model_name = f"{args.decoder}_{args.loss}"
        summary_path = Path(args.save_dir) / f'cv_summary_{model_name}.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved to: {summary_path}")


if __name__ == '__main__':
    main()
