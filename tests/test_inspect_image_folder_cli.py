import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from intrep import inspect_image_folder


class InspectImageFolderCLITest(unittest.TestCase):
    def test_prints_image_folder_summary(self) -> None:
        output = io.StringIO()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            apple_dir = root / "apple"
            zebra_dir = root / "zebra"
            apple_dir.mkdir()
            zebra_dir.mkdir()
            Image.new("RGB", (2, 3), color=(255, 0, 0)).save(apple_dir / "a.png")
            Image.new("RGB", (2, 3), color=(0, 255, 0)).save(zebra_dir / "z.png")

            with redirect_stdout(output):
                inspect_image_folder.main([str(root)])

        text = output.getvalue()
        self.assertIn("intrep image folder", text)
        self.assertIn("classes=2", text)
        self.assertIn("images=2", text)
        self.assertIn("image_shape=(3, 2, 3)", text)
        self.assertIn("channels=3", text)
        self.assertIn("first_classes=apple,zebra", text)

    def test_accepts_image_size_override(self) -> None:
        output = io.StringIO()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            class_dir = root / "apple"
            class_dir.mkdir()
            Image.new("RGB", (2, 3), color=(255, 0, 0)).save(class_dir / "a.png")

            with redirect_stdout(output):
                inspect_image_folder.main([str(root), "--image-size", "4", "5"])

        text = output.getvalue()
        self.assertIn("image_shape=(4, 5, 3)", text)
        self.assertIn("first_image_shape=(4, 5, 3)", text)


if __name__ == "__main__":
    unittest.main()
