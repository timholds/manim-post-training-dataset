"""Kaggle dataset extractors."""

import pandas as pd
from pathlib import Path
import logging
from typing import Iterator, Dict, Any, Optional

from ..base import BaseExtractor
from ..registry import register_extractor

logger = logging.getLogger(__name__)


@register_extractor
class ManimbenchExtractor(BaseExtractor):
    """Extractor for the Manimbench dataset from Kaggle."""
    
    source_id = "manimbench"
    source_name = "Manim SFT Dataset (Manimbench)"
    priority = 4  # Highest quality, reviewed descriptions
    
    def _validate_config(self) -> None:
        """Validate Kaggle extractor configuration."""
        self.kaggle_dataset = self.config.get("kaggle_dataset", "ravidussilva/manim-sft")
        self.file_name = self.config.get("file", "manim_sft_dataset.parquet")
        self.cache_dir = Path(self.config.get("cache_dir", Path.home() / ".cache" / "manim_datasets"))
    
    def estimate_sample_count(self) -> Optional[int]:
        """Return estimated number of samples."""
        return 400
    
    def _download_dataset(self) -> Optional[Path]:
        """Download dataset from Kaggle using kagglehub."""
        try:
            import kagglehub
            
            logger.info(f"Downloading Kaggle dataset: {self.kaggle_dataset}")
            dataset_path = kagglehub.dataset_download(self.kaggle_dataset)
            logger.info(f"Downloaded to: {dataset_path}")
            return Path(dataset_path)
            
        except ImportError:
            logger.error("kagglehub not installed. Install with: pip install kagglehub")
            return None
        except Exception as e:
            logger.error(f"Failed to download Kaggle dataset: {e}")
            return None
    
    def _load_dataframe(self) -> Optional[pd.DataFrame]:
        """Load the dataset as a pandas DataFrame."""
        # Check cache first
        cache_file = self.cache_dir / f"{self.kaggle_dataset.replace('/', '_')}_{self.file_name}"
        
        if cache_file.exists():
            logger.info(f"Loading cached dataset from {cache_file}")
            return pd.read_parquet(cache_file)
        
        # Download dataset
        download_path = self._download_dataset()
        if not download_path:
            return None
        
        # Find the parquet file
        parquet_file = download_path / self.file_name
        if not parquet_file.exists():
            parquet_files = list(download_path.glob("*.parquet"))
            if parquet_files:
                parquet_file = parquet_files[0]
                logger.info(f"Using parquet file: {parquet_file}")
            else:
                logger.error(f"No parquet file found in {download_path}")
                return None
        
        # Load and cache
        df = pd.read_parquet(parquet_file)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_file)
        logger.info(f"Cached dataset to {cache_file}")
        
        return df
    
    def extract(self) -> Iterator[Dict[str, Any]]:
        """Extract samples from Manimbench dataset."""
        df = self._load_dataframe()
        if df is None:
            logger.error("Failed to load Manimbench dataset")
            return
        
        logger.info(f"Processing {len(df)} samples from Manimbench")
        
        # Process each row
        for idx, row in df.iterrows():
            try:
                description = str(row.get("Reviewed Description", ""))
                code = str(row.get("Code", ""))
                
                # Skip invalid entries
                if not description or not code or description == 'nan' or code == 'nan':
                    continue
                
                yield {
                    "description": description,
                    "code": code,
                    "metadata": {
                        "split": row.get("Split", "train"),
                        "original_description": row.get("Original Description", ""),
                        "row_index": idx
                    }
                }
                
            except Exception as e:
                logger.warning(f"Error processing row {idx}: {e}")
                continue