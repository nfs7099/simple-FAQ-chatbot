import os
import sys
import argparse

try:
    from src.utils.logger import setup_logger
except Exception:
    import logging
    def setup_logger():
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        )
        return logging.getLogger("run")

from src.data_processing.process_pdf import main as process_main
from src.api.app import start

from configs.config import (
    HOST, PORT, DEBUG, PDF_DIR, VECTOR_DB_PATH,
    CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RESULTS,
    LLM_PROVIDER, OLLAMA_MODEL, OPENAI_MODEL, ANTHROPIC_MODEL,
    USE_TOKEN_SPLITTER, TOKENIZER_NAME, TOKEN_CHUNK_SIZE, TOKEN_CHUNK_OVERLAP,
    ALLOW_DANGEROUS_DESERIALIZATION
)
from configs.config import validate_configuration


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the Simple FAQ Chatbot application."
    )
    parser.add_argument(
        "action",
        choices=["process", "run", "all"],
        help="Action to perform: process (process PDFs), run (start server), all (process and run)",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Host to run the server on (overrides .env/config if provided)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to run the server on (overrides .env/config if provided)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Force DEBUG=True (uvicorn reload on) regardless of .env",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="After indexing, ask a test question to verify end-to-end RAG",
    )
    return parser.parse_args()


def banner(logger):
    model = (
        OLLAMA_MODEL if LLM_PROVIDER == "ollama"
        else OPENAI_MODEL if LLM_PROVIDER == "openai"
        else ANTHROPIC_MODEL if LLM_PROVIDER == "anthropic"
        else "unknown"
    )
    logger.info("========== Simple Chatbot ==========")
    logger.info(f"Provider     : {LLM_PROVIDER} | Model: {model}")
    logger.info(f"PDF dir      : {PDF_DIR}")
    logger.info(f"Vector DB    : {VECTOR_DB_PATH}")
    logger.info(f"Chunking     : chars size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")
    logger.info(f"FAISS pickle load : {'ENABLED' if ALLOW_DANGEROUS_DESERIALIZATION else 'DISABLED'}")
    logger.info(
        "Token split  : %s%s",
        "ON" if USE_TOKEN_SPLITTER else "OFF",
        f" (tokenizer={TOKENIZER_NAME}, size={TOKEN_CHUNK_SIZE or CHUNK_SIZE}, "
        f"overlap={TOKEN_CHUNK_OVERLAP or CHUNK_OVERLAP})" if USE_TOKEN_SPLITTER else "",
    )
    logger.info(f"Retriever k  : {TOP_K_RESULTS}")
    logger.info("========================================")


def process_pdfs(logger=None, smoke_test: bool = False) -> bool:
    if logger:
        logger.info("Processing PDF files...")
    try:
        if smoke_test:
            os.environ["PROCESS_SMOKE_TEST"] = "true"
        else:
            os.environ.pop("PROCESS_SMOKE_TEST", None)

        process_main()

        if logger:
            logger.info("PDF processing step finished.")
        return True
    except Exception as e:
        if logger:
            logger.error(f"Error processing PDFs: {e}", exc_info=True)
        return False


def run_server(host: str = None, port: int = None, debug: bool = False, logger=None) -> bool:
    if logger:
        logger.info("Starting server...")
    try:
        if host is not None:
            os.environ["HOST"] = host
        if port is not None:
            os.environ["PORT"] = str(port)
        if debug:
            os.environ["DEBUG"] = "true"

        start()
        if logger:
            logger.info("Server started successfully.")
        return True
    except Exception as e:
        if logger:
            logger.error(f"Error starting server: {e}", exc_info=True)
        return False


def main() -> int:
    logger = setup_logger()
    args = parse_args()

    errors = validate_configuration()
    if errors:
        logger.error("Configuration errors detected:")
        for err in errors:
            logger.error("  - %s", err)
        return 1

    banner(logger)
    logger.info("Action: %s", args.action)

    # 1) process PDFs if requested
    if args.action in ("process", "all"):
        ok = process_pdfs(logger=logger, smoke_test=args.smoke_test)
        if not ok:
            if args.action == "process":
                logger.error("PDF processing failed. Exiting.")
                return 1
            else:
                logger.warning("PDF processing failed. Continuing to server start...")

    # 2) start server if requested
    if args.action in ("run", "all"):
        ok = run_server(args.host, args.port, debug=args.debug, logger=logger)
        if not ok:
            logger.error("Server failed to start. Exiting.")
            return 1

    logger.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
