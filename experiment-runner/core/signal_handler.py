#!/usr/bin/env python3
"""
Signal handler module.

Handles system signals (SIGINT, SIGTERM) for graceful shutdown.
"""

import signal
import logging
import sys
from typing import Callable, List, Dict, Any

logger = logging.getLogger(__name__)

class SignalHandler:
    """
    Signal handler for graceful shutdown.
    
    Registers handlers for system signals and runs cleanup functions on shutdown.
    """
    
    def __init__(self):
        """Initialize signal handler."""
        self.cleanup_functions: List[Callable[[], None]] = []
        self.shutdown_initiated: bool = False
        
    def register_cleanup_function(self, func: Callable[[], None]) -> None:
        """
        Register a function to be called on shutdown.
        
        Args:
            func: Function to call during cleanup
        """
        if func not in self.cleanup_functions:
            self.cleanup_functions.append(func)
            logger.debug(f"Registered cleanup function: {func.__name__}")
        
    def handle_signal(self, signum: int, frame: Any) -> None:
        """
        Handle system signal.
        
        Args:
            signum: Signal number
            frame: Current stack frame
        """
        if self.shutdown_initiated:
            logger.warning("Received second signal, forcing exit...")
            sys.exit(1)
            
        self.shutdown_initiated = True
        signal_name = signal.Signals(signum).name
        
        logger.info(f"Received signal {signum} ({signal_name}). Initiating graceful shutdown...")
        
        # Run cleanup functions
        for func in reversed(self.cleanup_functions):
            try:
                logger.info(f"Running cleanup function: {func.__name__}")
                func()
            except Exception as e:
                logger.error(f"Error in cleanup function {func.__name__}: {e}")
        
        logger.info("Cleanup complete. Exiting...")
        sys.exit(0)
        
    def setup_signal_handlers(self) -> None:
        """
        Set up signal handlers for SIGINT and SIGTERM.
        """
        logger.info("Setting up signal handlers")
        signal.signal(signal.SIGINT, self.handle_signal)
        signal.signal(signal.SIGTERM, self.handle_signal)
