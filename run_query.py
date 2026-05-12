import logging
import shutil
import os

from argparse import ArgumentParser, Namespace
from dotenv import load_dotenv
from pathlib import Path

from hmr.app import ReaderRetrievalApp
from hmr.config import AppConfig, IngestionConfig, RetrievalConfig, StorageConfig
from hmr.logging_config import setup_logging
from hmr.llm.prompted_service import PromptedReaderLLMService
from hmr.llm.openai_compatible import OpenAICompatibleLLMClient


def parser_argment() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--query", action="append", help="Ask one query. Can be repeated.")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--runtime-dir", type=Path, default=Path("runtime"))
    parser.add_argument("--activation-threshold", type=float, default=0.08)
    parser.add_argument("--verbose", action="store_true", help="Enable DEBUG logs.")
    return parser.parse_args()


def build_config(args: Namespace) -> AppConfig:
    return AppConfig(
        ingestion=IngestionConfig(),
        retrieval=RetrievalConfig(top_k=args.top_k, activation_threshold=args.activation_threshold),
        storage=StorageConfig(
            sqlite_path=args.runtime_dir / "hmr_demo.sqlite3",
            chroma_path=args.runtime_dir / "chroma",
            chroma_collection="reader_capabilities",
        ),
    )


def main():
    args = parser_argment()
    load_dotenv()
    runtime_dir = args.runtime_dir
    runtime_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(
        log_path=runtime_dir / "query_demo.log",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )
    logging.getLogger(__name__).info("Runtime directory: %s", runtime_dir)
    
    config = build_config(args)
    client = OpenAICompatibleLLMClient(
        api_key=os.environ.get("API_KEY", ""),
        model="deepseek-v4-flash",
        base_url="https://api.deepseek.com"
    )
    llm_service = PromptedReaderLLMService(client)
    app = ReaderRetrievalApp(config, llm_service=llm_service)
    for query in args.query:
        result = app.ask(query)
        print("=" * 88)
        print(result.answer)
        print("\nActivated Readers:")
        for answer in result.activated_answers:
            print(f"- {answer.title} | confidence={answer.confidence:.2f}")
        print() 


if __name__ == "__main__":
    main()