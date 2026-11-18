"""TGMR-Det reference research implementation (thermal-reference detection)."""
from .model import TGMRDet, ITSM, MSFRNet, SSFCFM
from .data import PairedVisibleThermalDataset, paired_collate_fn
from .losses import DetectionLoss
