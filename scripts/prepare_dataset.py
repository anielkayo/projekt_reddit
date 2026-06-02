from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.reddit_pbl.preparation import load_dataset, prepare_dataset, write_prepared_artifacts


DATA_PATH = Path("database/NajnowszaWersjaBazy1205.csv")
PREPARED_DATA_PATH = Path("database/NajnowszaWersjaBazy1205_prepared.csv")
OUTPUT_DIR = Path("outputs")


def main() -> None:
    raw = load_dataset(DATA_PATH)
    prepared = prepare_dataset(raw)
    paths = write_prepared_artifacts(
        prepared,
        prepared_dataset_path=PREPARED_DATA_PATH,
        output_dir=OUTPUT_DIR,
    )

    print(f"Prepared dataset: {paths['prepared_dataset']}")
    print(f"Quality report: {paths['quality_report_csv']}")
    print(f"Metadata: {paths['metadata_json']}")
    print(f"Markdown report: {paths['markdown_report']}")


if __name__ == "__main__":
    main()
