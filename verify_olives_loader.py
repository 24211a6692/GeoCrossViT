"""
verify_olives_loader.py

This script verifies the correct implementation of the OlivesDataset and RetinalDataModule
for the OLIVES dataset (Prime_FULL cohort). It checks sample pairing, labeling,
class distribution, dataset split exclusivity, and dataloader compatibility.
"""

import sys
import torch
from collections import Counter

from configs.config import GeoCrossViTConfig
from datasets.datamodule import RetinalDataModule


def verify_all():
    print("=== STARTING OLIVES DATASET VERIFICATION ===")
    
    # 1. Initialize configuration and DataModule
    config = GeoCrossViTConfig()
    data_module = RetinalDataModule(config)
    
    print("\n--- Setting up Datasets ---")
    data_module.setup(stage=None)
    
    train_dataset = data_module.train_dataset
    val_dataset = data_module.val_dataset
    test_dataset = data_module.test_dataset
    
    assert train_dataset is not None, "Train dataset was not instantiated."
    assert val_dataset is not None, "Validation dataset was not instantiated."
    assert test_dataset is not None, "Test dataset was not instantiated."
    
    # Check skipped reasons
    print(f"Skipped samples during building:")
    for reason, count in train_dataset.skipped_counts.items():
        print(f"  - {reason}: {count}")
    
    # 2. Check total valid paired samples
    all_valid = train_dataset.all_valid_samples
    total_paired = len(all_valid)
    print(f"\nTotal number of paired samples: {total_paired}")
    assert total_paired > 0, "No valid paired samples were found."
    
    # 3. Verify Fundus-OCT pairing and valid DRSS label for every sample
    print("\nVerifying each valid sample...")
    for idx, sample in enumerate(all_valid):
        fundus_p = sample["fundus_path"]
        oct_p = sample["oct_path"]
        label = sample["label"]
        
        # Verify both paths exist
        assert fundus_p is not None and len(fundus_p) > 0, f"Sample {idx} has invalid fundus path."
        assert oct_p is not None and len(oct_p) > 0, f"Sample {idx} has invalid OCT path."
        
        # Verify label is valid class [0, 4]
        assert label in [0, 1, 2, 3, 4], f"Sample {idx} has invalid DRSS mapped class label: {label}."
    
    print("Verification passed: Every Fundus image has a paired OCT image, and every paired OCT image has a valid DRSS label.")
    
    # 4. Class distribution verification
    labels = [sample["label"] for sample in all_valid]
    label_counts = Counter(labels)
    
    print("\nDRSS Class distribution (0-4) across all paired samples:")
    for class_idx in sorted(label_counts.keys()):
        print(f"  - Class {class_idx}: {label_counts[class_idx]} samples ({label_counts[class_idx] / total_paired * 100:.2f}%)")
        
    # Verify all 5 classes are present
    for class_idx in range(5):
        assert label_counts[class_idx] > 0, f"Class {class_idx} is missing from the dataset!"
    print("Verification passed: All five classes are present in the dataset.")
    
    # 5. Patient-level split exclusivity verification
    train_pids = set(sample["patient_id"] for sample in train_dataset.samples)
    val_pids = set(sample["patient_id"] for sample in val_dataset.samples)
    test_pids = set(sample["patient_id"] for sample in test_dataset.samples)
    
    print(f"\nDataset Splits:")
    print(f"  - Train: {len(train_dataset)} samples, {len(train_pids)} unique patients")
    print(f"  - Val:   {len(val_dataset)} samples, {len(val_pids)} unique patients")
    print(f"  - Test:  {len(test_dataset)} samples, {len(test_pids)} unique patients")
    
    # Assert mutual exclusivity
    train_val_overlap = train_pids & val_pids
    train_test_overlap = train_pids & test_pids
    val_test_overlap = val_pids & test_pids
    
    print(f"\nExclusivity Overlaps:")
    print(f"  - Train & Val overlap:  {len(train_val_overlap)} patients")
    print(f"  - Train & Test overlap: {len(train_test_overlap)} patients")
    print(f"  - Val & Test overlap:   {len(val_test_overlap)} patients")
    
    assert len(train_val_overlap) == 0, f"Train and Val splits share patient IDs: {train_val_overlap}"
    assert len(train_test_overlap) == 0, f"Train and Test splits share patient IDs: {train_test_overlap}"
    assert len(val_test_overlap) == 0, f"Val and Test splits share patient IDs: {val_test_overlap}"
    print("Verification passed: Train, validation, and test patient IDs are mutually exclusive.")
    
    # 6. Dataloader compatibility verification
    print("\nTesting Dataloader batch retrieval...")
    train_loader = data_module.train_dataloader()
    first_batch = next(iter(train_loader))
    
    required_keys = {"fundus", "oct", "label", "patient_id", "sample_id"}
    batch_keys = set(first_batch.keys())
    assert required_keys.issubset(batch_keys), f"Batch missing required keys. Expected {required_keys}, got {batch_keys}"
    
    fundus = first_batch["fundus"]
    oct_img = first_batch["oct"]
    batch_labels = first_batch["label"]
    
    print(f"  - Batch retrieved successfully!")
    print(f"  - Fundus tensor shape: {fundus.shape} (Expected: [batch_size, 3, 224, 224])")
    print(f"  - OCT tensor shape:    {oct_img.shape} (Expected: [batch_size, 3, 224, 224])")
    print(f"  - Label tensor shape:  {batch_labels.shape} (Expected: [batch_size])")
    print(f"  - Label data type:     {batch_labels.dtype} (Expected: torch.long)")
    
    assert fundus.ndim == 4 and fundus.shape[1:] == (3, 224, 224), f"Incorrect fundus shape: {fundus.shape}"
    assert oct_img.ndim == 4 and oct_img.shape[1:] == (3, 224, 224), f"Incorrect OCT shape: {oct_img.shape}"
    assert batch_labels.ndim == 1, f"Incorrect label shape: {batch_labels.shape}"
    assert batch_labels.dtype == torch.long, f"Incorrect label dtype: {batch_labels.dtype}"
    
    print("\n=== ALL VERIFICATIONS PASSED SUCCESSFULLY! ===")


if __name__ == "__main__":
    verify_all()
