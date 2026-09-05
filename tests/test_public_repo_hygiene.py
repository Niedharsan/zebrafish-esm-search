import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PublicRepoHygieneTests(unittest.TestCase):
    def test_macos_launchers_do_not_embed_developer_home_paths(self) -> None:
        paths = [
            ROOT / "start_dashboard.command",
            ROOT / "Zebrafish ESM Dashboard.app/Contents/MacOS/Zebrafish ESM Dashboard",
        ]
        combined = "\n".join(path.read_text(encoding="utf-8") for path in paths)

        self.assertNotIn("/Users/niedharsan/", combined)
        self.assertIn('dirname "$0"', combined)

    def test_unused_generated_root_image_is_not_committed(self) -> None:
        self.assertFalse((ROOT / "ChatGPT Image Jun 15, 2026, 01_30_30 PM.png").exists())


if __name__ == "__main__":
    unittest.main()
