"""
datasets/olives_dataset.py

This module contains the OlivesDataset class which handlespaired loading of Fundus
and OCT scan images, clinical metadata, and DRSS classification targets.
"""

import os
import re
import hashlib
from typing import Dict, List, Tuple, Optional, Any
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset

from configs.config import GeoCrossViTConfig
from datasets.transforms import CoordinatedGeoTransforms


class OlivesDataset(Dataset):
    """
    OLIVES Dataset implementation for Prime_FULL cohort.
    Pairs 2D Fundus photography with cross-sectional OCT B-scan slices,
    extracts clinical annotations, maps DRSS scores to 5 classification targets,
    and partitions the dataset deterministically on patient level.
    """
    def __init__(
        self,
        config: GeoCrossViTConfig,
        transform: CoordinatedGeoTransforms,
        split: str = "train"
    ) -> None:
        """
        Args:
            config (GeoCrossViTConfig): Global config configuration.
            transform (CoordinatedGeoTransforms): Coordinated spatial transformations.
            split (str): One of 'train', 'val', or 'test'.
        """
        self.config = config
        self.transform = transform
        self.split = split.lower()
        
        # Load paths from config
        self.base_dir = "D:/dataset/OLIVES/OLIVES"
        self.xlsx_path = config.dataset.xlsx_path
        self.csv_path = config.dataset.csv_path
        self.dr_xlsx_path = config.dataset.dr_xlsx_path

        # 1. Parse DRSS lookup map
        self.drss_lookup = self._parse_drss_lookup()

        # 2. Parse biomarker metadata from CSV
        self.biomarker_lookup = self._parse_biomarker_lookup()

        # 3. Build paired samples in memory
        self.all_valid_samples, self.skipped_counts = self._build_paired_samples()

        # 4. Partition dataset by Patient_ID deterministically
        self.samples = self._partition_samples()

    def _normalize_path(self, path: str) -> str:
        """
        Normalizes paths by converting slashes, case, and stripping file extensions.
        """
        p = str(path).strip().replace("\\", "/")
        p = "/" + p.lstrip("/")
        p = p.lower()
        p_no_ext, _ = os.path.splitext(p)
        return p_no_ext

    def _parse_drss_lookup(self) -> Dict[Tuple[str, str, str], Any]:
        """
        Parses DRSS sheet in OCT-DR.xlsx and returns lookup dict: (patient_id, eye, visit) -> DRSS score
        """
        df_drss = pd.read_excel(self.dr_xlsx_path, sheet_name="DRSS")
        header_row = df_drss.iloc[0]
        data_rows = df_drss.iloc[1:]

        visit_cols = {}
        for col_idx, (col_name, val) in enumerate(zip(df_drss.columns, header_row)):
            if val == "DRSS Level":
                visit_str = str(col_name).strip()
                if visit_str == "Screen" or visit_str == "Baseline":
                    visit_key = "W0"
                else:
                    m = re.search(r"Week\s*(\d+)", visit_str, re.IGNORECASE)
                    if m:
                        visit_key = f"W{m.group(1)}"
                    else:
                        visit_key = visit_str
                visit_cols[col_idx] = visit_key

        lookup = {}
        for _, row in data_rows.iterrows():
            pid = str(row.iloc[0]).strip()
            eye = str(row.iloc[3]).strip().upper()  # Column Index 3 is Eye ID (OS/OD)
            for col_idx, visit_key in visit_cols.items():
                drss_val = row.iloc[col_idx]
                if pd.notna(drss_val):
                    lookup[(pid, eye, visit_key)] = drss_val
        return lookup

    def _parse_biomarker_lookup(self) -> Dict[str, Tuple[List[int], float, float]]:
        """
        Parses Biomarker_Clinical_Data_Images.csv and builds path to (biomarkers, BCVA, CST) lookup map.
        """
        df_csv = pd.read_csv(self.csv_path)
        # Keep only Prime_FULL rows for efficiency
        df_prime_csv = df_csv[df_csv["Path (Trial/Arm/Folder/Visit/Eye/Image Name)"].str.contains("Prime_FULL", case=False, na=False)].copy()

        biomarker_cols = [
            'Atrophy / thinning of retinal layers', 'Disruption of EZ', 'DRIL', 'IR hemorrhages', 'IR HRF',
            'Partially attached vitreous face', 'Fully attached vitreous face', 'Preretinal tissue/hemorrhage',
            'Vitreous debris', 'VMT', 'DRT/ME', 'Fluid (IRF)', 'Fluid (SRF)', 'Disruption of RPE',
            'PED (serous)', 'SHRM'
        ]

        lookup = {}
        for _, row in df_prime_csv.iterrows():
            path_key = str(row["Path (Trial/Arm/Folder/Visit/Eye/Image Name)"]).strip()
            norm_key = self._normalize_path(path_key)
            
            # Extract biomarkers
            b_list = []
            for col in biomarker_cols:
                try:
                    val = int(float(row[col]))
                except (ValueError, TypeError):
                    val = 0
                b_list.append(val)
                
            # Extract BCVA and CST
            try:
                bcva = float(row["BCVA"])
            except (ValueError, TypeError):
                bcva = -1.0
            try:
                cst = float(row["CST"])
            except (ValueError, TypeError):
                cst = -1.0
                
            lookup[norm_key] = (b_list, bcva, cst)
        return lookup

    def _map_drss_to_class(self, drss_val: Any) -> Optional[int]:
        """
        Maps raw DRSS value to class index 0-4. Returns None if invalid.
        """
        try:
            val = int(float(drss_val))
        except (ValueError, TypeError):
            return None
        
        if val < 35:
            return 0  # No DR (fallback)
        elif val == 35:
            return 0  # DRSS 35 (mildest class in Prime_FULL)
        elif val == 43:
            return 1  # DRSS 43 (next mild class in Prime_FULL)
        elif val in [47, 53]:
            return 2  # Moderate NPDR (DRSS 47, 53)
        elif val in [61, 65]:
            return 3  # Severe NPDR (DRSS 61, 65)
        elif val >= 71:
            return 4  # PDR / Advanced PDR (DRSS 71, 75, 81, 85)
        return None

    def _build_paired_samples(self) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """
        Builds all Fundus-OCT B-scan pairs from disk and Excel registry.
        Caches Fundus location per directory to maximize build speed.
        """
        df_images = pd.read_excel(self.xlsx_path)
        df_prime = df_images[df_images["File_Path"].str.contains("Prime_FULL", case=False, na=False)].copy()

        skipped_counts = {
            "missing_drss_record": 0,
            "invalid_drss_value": 0,
            "missing_oct_file": 0,
            "missing_fundus_file": 0,
            "missing_biomarker_record": 0
        }

        all_valid_samples = []
        fundus_dir_cache = {}  # parent_dir -> absolute_fundus_path

        for _, row in df_prime.iterrows():
            rel_path = str(row["File_Path"]).strip()
            
            # Parse Patient, Visit, Eye from file path
            # Path format: /Prime_FULL/01-001/W0/OS/27.png
            parts = rel_path.strip("/").split("/")
            if len(parts) < 5:
                continue
            
            patient_id = parts[1]
            visit = parts[2]
            eye = parts[3].upper()

            # 1. Look up DRSS score
            drss_key = (patient_id, eye, visit)
            if drss_key not in self.drss_lookup:
                skipped_counts["missing_drss_record"] += 1
                continue

            drss_val = self.drss_lookup[drss_key]
            label = self._map_drss_to_class(drss_val)
            if label is None:
                skipped_counts["invalid_drss_value"] += 1
                continue

            # 2. Look up biomarkers, BCVA, CST using normalized paths
            norm_path = self._normalize_path(rel_path)
            if norm_path not in self.biomarker_lookup:
                skipped_counts["missing_biomarker_record"] += 1
                continue
            biomarkers, bcva, cst = self.biomarker_lookup[norm_path]

            # 3. Check OCT file existence on disk (trying original and alternative extensions)
            # Remove redundant trial prefix and join with img_dir config to handle Prime_FULL/Prime_FULL
            sub_path = rel_path
            if sub_path.lower().startswith("/prime_full/"):
                sub_path = sub_path[len("/prime_full/"):]
            elif sub_path.lower().startswith("prime_full/"):
                sub_path = sub_path[len("prime_full/"):]
                
            oct_abs = os.path.join(self.config.dataset.img_dir, sub_path.lstrip("/"))
            if not os.path.exists(oct_abs):
                root, ext = os.path.splitext(oct_abs)
                alt_ext = ".png" if ext.lower() == ".tif" else ".tif"
                oct_abs_alt = root + alt_ext
                if os.path.exists(oct_abs_alt):
                    oct_abs = oct_abs_alt
                else:
                    skipped_counts["missing_oct_file"] += 1
                    continue

            # 4. Check/Find Fundus file in directory
            parent_dir = os.path.dirname(oct_abs)
            if parent_dir not in fundus_dir_cache:
                fundus_path = None
                if os.path.exists(parent_dir):
                    for filename in os.listdir(parent_dir):
                        if "fundus" in filename.lower():
                            fundus_path = os.path.join(parent_dir, filename)
                            break
                fundus_dir_cache[parent_dir] = fundus_path

            fundus_abs = fundus_dir_cache[parent_dir]
            if fundus_abs is None or not os.path.exists(fundus_abs):
                skipped_counts["missing_fundus_file"] += 1
                continue

            # Generate stable sample ID
            sample_name = os.path.splitext(os.path.basename(oct_abs))[0]
            sample_id = f"{patient_id}_{visit}_{eye}_{sample_name}"

            all_valid_samples.append({
                "fundus_path": fundus_abs,
                "oct_path": oct_abs,
                "label": label,
                "patient_id": patient_id,
                "sample_id": sample_id,
                "bcva": bcva,
                "cst": cst,
                "biomarkers": biomarkers
            })

        return all_valid_samples, skipped_counts

    def _partition_samples(self) -> List[Dict[str, Any]]:
        """
        Partitions the samples list into the target split using a deterministic hash on patient IDs.
        """
        split_samples = []
        for sample in self.all_valid_samples:
            pid = sample["patient_id"]
            # Deterministic patient-level split (70/15/15)
            h = int(hashlib.sha256(pid.encode()).hexdigest(), 16)
            val = h % 100
            
            if val < 70:
                p_split = "train"
            elif val < 85:
                p_split = "val"
            else:
                p_split = "test"
                
            if p_split == self.split:
                split_samples.append(sample)
                
        return split_samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        
        # Load images
        fundus_img = Image.open(sample["fundus_path"]).convert("RGB")
        oct_img = Image.open(sample["oct_path"]).convert("RGB")
        
        # Apply coordinated transforms
        fundus_tensor, oct_tensor = self.transform(fundus_img, oct_img)
        
        return {
            "fundus": fundus_tensor,
            "oct": oct_tensor,
            "label": torch.tensor(sample["label"], dtype=torch.long),
            "patient_id": sample["patient_id"],
            "sample_id": sample["sample_id"]
        }
