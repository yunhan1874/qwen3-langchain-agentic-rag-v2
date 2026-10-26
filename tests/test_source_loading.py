from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rag_v2.semantic_chunking import load_source_documents


class SourceLoadingTests(unittest.TestCase):
    def test_multiline_medical_records_are_reconstructed_and_deduplicated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "docs.txt"
            path.write_text(
                "问题: 高血压？\n答案: 控制血压。\n补充说明。\n"
                "问题: 糖尿病？\n答案: 监测血糖。\n"
                "问题: 高血压？\n答案: 控制血压。\n补充说明。\n",
                encoding="utf-8",
            )
            documents = load_source_documents(path)
        self.assertEqual(2, len(documents))
        self.assertIn("补充说明", documents[0].text)
        self.assertEqual(1, documents[0].metadata["source_line"])
        self.assertEqual(4, documents[1].metadata["source_line"])


if __name__ == "__main__":
    unittest.main()
