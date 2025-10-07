import logging
import os
from datetime import datetime

class Logger:
    """Logging utility for FACCP"""
    
    def __init__(self, config):
        self.config = config
        self.setup_logger()
    
    def setup_logger(self):
        """Setup logging configuration"""
        logs_dir = os.path.join(self.config.logs_dir, 
                               datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(logs_dir, exist_ok=True)
        
        log_file = os.path.join(logs_dir, "faccp.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger('FACCP')
    
    def info(self, message):
        self.logger.info(message)
    
    def warning(self, message):
        self.logger.warning(message)
    
    def error(self, message):
        self.logger.error(message)
    
    def debug(self, message):
        self.logger.debug(message)
    
    # New: Log UQ metrics
    def log_uq(self, uncertainty: float, layer_name: str):
        self.logger.info(f"UQ - Layer {layer_name}: Average Uncertainty = {uncertainty:.6f}")