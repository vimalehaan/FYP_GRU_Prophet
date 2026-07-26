"""Helpers for visualization notebooks — embed figures in saved .ipynb outputs."""

from __future__ import annotations

from pathlib import Path


def show_png_gallery(
    directory: str | Path,
    *,
    fig_width: float = 10.0,
) -> None:
    """
    Display pre-exported PNG files inline (embedded in notebook when executed).

    Prefer this over ``IPython.display.Image(filename=...)``, which often saves
    path references that do not render when reopening the notebook.
    """
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    plot_dir = Path(directory)
    paths = sorted(plot_dir.glob("*.png"))
    if not paths:
        print(f"No PNG files in {plot_dir}")
        return

    for path in paths:
        print(path.name)
        img = mpimg.imread(path)
        height, width = img.shape[:2]
        fig_height = max(3.0, fig_width * height / max(width, 1))
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(path.stem.replace("_", " "))
        plt.tight_layout()
        plt.show()
