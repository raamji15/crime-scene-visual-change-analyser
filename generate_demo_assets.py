"""Generate non-graphic, reproducible synthetic evidence-scene demo images."""

from pathlib import Path

import cv2
import numpy as np


ASSET_DIRECTORY = Path(__file__).parent / "assets"


def make_scene(with_changes: bool) -> np.ndarray:
    image = np.full((600, 900, 3), (236, 232, 222), dtype=np.uint8)
    cv2.rectangle(image, (40, 55), (860, 540), (190, 202, 211), -1)
    cv2.rectangle(image, (75, 95), (825, 500), (220, 216, 203), -1)
    cv2.line(image, (75, 410), (825, 410), (135, 130, 120), 5)
    cv2.rectangle(image, (120, 155), (280, 370), (105, 135, 158), -1)
    cv2.rectangle(image, (133, 168), (267, 357), (163, 185, 196), -1)
    cv2.rectangle(image, (540, 150), (735, 315), (91, 85, 78), -1)
    cv2.rectangle(image, (553, 163), (722, 302), (197, 176, 138), -1)
    cv2.rectangle(image, (390, 350), (575, 410), (111, 84, 63), -1)
    cv2.rectangle(image, (410, 325), (550, 355), (138, 109, 81), -1)
    cv2.circle(image, (485, 340), 20, (68, 110, 68), -1)
    cv2.putText(image, "SYNTHETIC DEMO SCENE", (250, 475), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (75, 75, 75), 2)
    if with_changes:
        cv2.rectangle(image, (650, 362), (735, 425), (34, 108, 184), -1)
        cv2.rectangle(image, (656, 368), (729, 419), (71, 150, 220), -1)
        cv2.circle(image, (205, 255), 23, (34, 34, 215), -1)
        cv2.line(image, (420, 385), (548, 385), (40, 40, 40), 6)
    return image


def main() -> None:
    ASSET_DIRECTORY.mkdir(exist_ok=True)
    cv2.imwrite(str(ASSET_DIRECTORY / "demo_before.png"), make_scene(False))
    cv2.imwrite(str(ASSET_DIRECTORY / "demo_after.png"), make_scene(True))
    print(f"Wrote demo assets to {ASSET_DIRECTORY}")


if __name__ == "__main__":
    main()
