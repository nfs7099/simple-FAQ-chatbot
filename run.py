import os
import sys
import logging
import argparse
import logging

from src.utils.logger import setup_logger
from src.data_processing.process_pdf import main as process_main
from src.api.app import start


logger=logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the Simple FAQ Chatbot application.")

    parser.add_argument(
        'action', 
        choices=['process', 'run', 'all'], 
        help='Action to perform: process (process PDFs), run (start server), all(process and run)'
    )
   
    parser.add_argument(
        '--host', 
        default=None, 
        help='Host to run the server on (overrides .env setting if provided)'
    )

    parser.add_argument(
        '--port', 
        default=None, 
        help='Port to run the server on (overrides .env setting if provided)'
    )

    return parser.parse_args()

def process_pdfs() -> bool:
    logger.info("Processing PDF files...")

    try:
        process_main()
        return True
    except Exception as e:
        logger.error(f"Error processing PDFs: {e}", exec_info=True)
        return False
    
def run_server(host: str = None, port: int = None) -> bool:
    logger.info("Starting server...")
    try:
        if host or port:
            from configs.config import HOST as default_host, PORT as default_port

            if host:
                os.environ['HOST'] = host
            if port:
                os.environ['PORT'] = str(port)
            
            logger.info(f"Server will run on host: {os.getenv('HOST', default_host)}, port: {os.getenv('PORT', default_port)}")

        start()
        return True
    except Exception as e:
        logger.error(f"Error starting server: {e}", exec_info=True)
        return False

def main():
    #centralized logging
    logger = setup_logger()
    logger.info("Application started")

    args = parse_args()
    if args.action in ['process', 'all']:
        success = process_pdfs()
        if not success and args.action == 'all':
            logger.warning("PDF processing failed. But continuing with server start...")
    
    if args.action in ['run', 'all']:
        success = run_server(args.host, args.port)
        if not success:
            logger.error("Server failed to start. Exiting application.")
            sys.exit(1)

if __name__ == "__main__":
    main()
