from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

from hmr.app import ReaderRetrievalApp
from hmr.config import AppConfig, IngestionConfig, RetrievalConfig, StorageConfig
from hmr.logging_config import setup_logging

DEFAULT_QUERIES = [
    "这个系统为什么不是寻找相似文本，而是寻找专家 Reader？",
    "查询阶段的粗召回和自评估激活分别做什么？",
    "SQLite 和 Chroma 在这个 demo 里分别负责什么？",
]


def main() -> None:
    args = parse_args()
    runtime_dir = args.runtime_dir
    if args.reset and runtime_dir.exists():
        shutil.rmtree(runtime_dir)
    runtime_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(
        log_path=runtime_dir / "hmr_demo.log",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )
    logging.getLogger(__name__).info("Runtime directory: %s", runtime_dir)

    config = build_config(args)
    app = ReaderRetrievalApp(config)
    try:
        root_id = app.ingest_file(args.doc, document_id=args.document_id)
        print(f"\n✅ Ingested document: {args.doc}")
        print(f"🌳 Root Reader: {root_id}\n")
        for query in args.query or DEFAULT_QUERIES:
            result = app.ask(query)
            print("=" * 88)
            print(result.answer)
            print("\nActivated Readers:")
            for answer in result.activated_answers:
                print(f"- {answer.title} | confidence={answer.confidence:.2f}")
            print()
        print(f"📄 Log file: {runtime_dir / 'hmr_demo.log'}")
        print(f"🗄️ SQLite DB: {config.storage.sqlite_path}")
        print(f"🧭 Chroma path: {config.storage.chroma_path}")
    finally:
        app.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick start for HMR demo")
    parser.add_argument("--doc", type=Path, default=Path("sample_docs/idea_demo.md"))
    parser.add_argument("--document-id", default="idea-demo")
    parser.add_argument("--runtime-dir", type=Path, default=Path("runtime"))
    parser.add_argument("--query", action="append", help="Ask one query. Can be repeated.")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--activation-threshold", type=float, default=0.08)
    parser.add_argument("--max-leaf-chars", type=int, default=900)
    parser.add_argument("--reset", action="store_true", help="Clear runtime storage before running.")
    parser.add_argument("--verbose", action="store_true", help="Enable DEBUG logs.")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> AppConfig:
    return AppConfig(
        ingestion=IngestionConfig(max_leaf_chars=args.max_leaf_chars),
        retrieval=RetrievalConfig(top_k=args.top_k, activation_threshold=args.activation_threshold),
        storage=StorageConfig(
            sqlite_path=args.runtime_dir / "hmr_demo.sqlite3",
            chroma_path=args.runtime_dir / "chroma",
            chroma_collection="reader_capabilities",
        ),
    )


if __name__ == "__main__":
    main()
