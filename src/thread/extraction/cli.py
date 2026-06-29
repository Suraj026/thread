"""thread-extract: One-shot entity extraction CLI.

Extracts SDLC entities from natural language text via stdin or file.

Usage:
    echo "We decided to use Redis for caching" | thread-extract
    thread-extract --file session.txt --format pretty
    thread-extract --group planning --threshold 0.5 --model openrouter:qwen3-coder:free
    thread-extract --version
"""

import argparse
import json
import sys
from typing import Optional

from thread.extraction.batch_extractor import BatchExtractor, BatchResult
from thread.extraction.config import Settings

VERSION = "0.1.0"


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:]).

    Returns:
        Parsed namespace with all option values.
    """
    parser = argparse.ArgumentParser(
        prog="thread-extract",
        description="Extract SDLC entities (goals, decisions, incidents, etc.) "
        "from natural language text.",
        epilog="Examples:\n"
        "  echo 'goal: reduce latency' | thread-extract\n"
        "  thread-extract --file session.log --format pretty\n"
        "  thread-extract --group planning --threshold 0.5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="OpenRouter model override (default: from settings/THREAD_OPENROUTER_MODEL_NAME)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Confidence threshold (0.0-1.0, default: 0.3)",
    )
    parser.add_argument(
        "--group",
        type=str,
        default=None,
        choices=["planning", "development", "operations", "incident", "learning", "collaboration"],
        help="Extractor group filter — run only one group",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="json",
        choices=["json", "pretty"],
        help="Output format (default: json — compact, pretty — indented)",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Read input from file instead of stdin",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version and exit",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress stderr progress messages",
    )

    return parser.parse_args(argv)


def format_output(result: BatchResult, fmt: str = "json") -> str:
    """Format a BatchResult as JSON string.

    Args:
        result: The batch extraction result to format.
        fmt: Output format — "json" (compact) or "pretty" (indented).

    Returns:
        JSON string representation of the result.
    """
    output = {
        "entities": [entity.model_dump() for entity in result.entities],
        "metadata": {
            "extracted_at": result.extracted_at.isoformat(),
            "entity_count": result.entity_count,
            "input_length": result.input_length,
        },
    }

    indent = 2 if fmt == "pretty" else None
    return json.dumps(output, indent=indent, default=str, ensure_ascii=False)


def main(argv: Optional[list[str]] = None) -> int:
    """Main entry point for the CLI tool.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:]).

    Returns:
        Exit code: 0 on success, 1 on failure.
    """
    args = parse_args(argv)

    # Handle --version early
    if args.version:
        print(f"thread-extract v{VERSION}")
        return 0

    # Read input text
    if args.file:
        try:
            with open(args.file, "r") as f:
                text = f.read()
        except FileNotFoundError:
            print(f"Error: file not found: {args.file}", file=sys.stderr)
            return 1
        except IOError as e:
            print(f"Error reading file: {e}", file=sys.stderr)
            return 1
    else:
        text = sys.stdin.read()

    if not text.strip():
        print("Error: no input text provided (stdin is empty)", file=sys.stderr)
        if not args.quiet:
            print("Hint: pipe text or use --file PATH", file=sys.stderr)
        return 1

    # Build settings with optional overrides
    settings_kwargs = {}
    if args.model:
        settings_kwargs["openrouter_model_name"] = args.model
    if args.threshold is not None:
        settings_kwargs["extraction_confidence_threshold"] = args.threshold

    settings = Settings(**settings_kwargs) if settings_kwargs else Settings()
    extractor = BatchExtractor(settings=settings)

    # Run extraction
    try:
        if args.group:
            entities = extractor.run_on_extractor_group(text, args.group)
            from thread.extraction.batch_extractor import BatchResult
            from datetime import datetime
            result = BatchResult(
                entities=entities,
                extracted_at=datetime.now(),
                input_length=len(text),
            )
        else:
            result = extractor.run(text)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Extraction failed: {e}", file=sys.stderr)
        return 1

    # Output results
    output = format_output(result, args.format)
    print(output)

    if not args.quiet:
        summary = extractor.entity_summary(result)
        print(
            f"  → {summary['total_entities']} entities extracted "
            f"(avg confidence: {summary['avg_confidence']})",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())