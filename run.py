import os
import sys
import argparse

from src.utils.logger import setup_logger
from src.data_processing.process_pdf import main as process_main
from src.api.app import start  # expects HOST/PORT from configs/env

def parse_args():
    parser = argparse.ArgumentParser(description="Run the Simple FAQ Chatbot application.")
    parser.add_argument(
        'action',
        choices=['process', 'run', 'all'],
        help='Action to perform: process (process PDFs), run (start server), all (process and run)'
    )
    parser.add_argument(
        '--host',
        default=None,
        help='Host to run the server on (overrides .env/config if provided)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=None,
        help='Port to run the server on (overrides .env/config if provided)'
    )
    return parser.parse_args()

def process_pdfs(logger=None) -> bool:
    if logger:
        logger.info("Processing PDF files...")
    try:
        process_main()
        if logger:
            logger.info("PDF processing completed successfully.")
        return True
    except Exception as e:
        if logger:
            logger.error(f"Error processing PDFs: {e}", exc_info=True)
        return False

def run_server(host: str = None, port: int = None, logger=None) -> bool:
    if logger:
        logger.info("Starting server...")
    try:
        # If CLI host/port provided, set them for the app to read (e.g., via configs/env)
        if host is not None:
            os.environ['HOST'] = host
        if port is not None:
            os.environ['PORT'] = str(port)

        start()
        if logger:
            logger.info("Server started successfully.")
        return True
    except Exception as e:
        if logger:
            logger.error(f"Error starting server: {e}", exc_info=True)
        return False

def main() -> int:
    """
    Entry point for the Simple FAQ Chatbot application.
    Parses arguments, sets up logging, optionally processes PDFs,
    and optionally starts the server.
    """
    logger = setup_logger()
    args = parse_args()

    logger.info("Application started with action=%s", args.action)

    # 1) Process PDFs if requested
    if args.action in ('process', 'all'):
        ok = process_pdfs(logger=logger)
        if not ok:
            if args.action == 'process':
                logger.error("PDF processing failed. Exiting.")
                return 1
            else:
                logger.warning("PDF processing failed. Continuing with server start...")

    # 2) Start server if requested
    if args.action in ('run', 'all'):
        ok = run_server(args.host, args.port, logger=logger)
        if not ok:
            logger.error("Server failed to start. Exiting.")
            return 1

    logger.info("Application finished.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
