import logging
from datetime import datetime
import os

class ProcessingLogger:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self._setup_logger()
        
    def _setup_logger(self):
        """Initial setup for the logging system"""
        log_file = f"{self.log_dir}/drilling_processor_{datetime.now().strftime('%Y%m%d')}.log"
        
        self.logger = logging.getLogger('DrillingProcessor')
        self.logger.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
    
    def log_processing_step(self, message: str, level: str = "info"):
        """Log a processing step"""
        if level.lower() == "info":
            self.logger.info(message)
        elif level.lower() == "warning":
            self.logger.warning(message)
        elif level.lower() == "error":
            self.logger.error(message)
