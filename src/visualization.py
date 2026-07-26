"""
==============================================================
VisionPro Studio
Visualization Module
==============================================================

Generates professional visualization dashboards,
comparison figures, histogram plots, and color palettes.

Uses a modern dark theme for all plots.

Author : VisionPro Studio Team
"""

from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams


# ==============================================================
# Global Theme Setup
# ==============================================================

def _apply_dark_theme():
    """
    Configure matplotlib with a professional dark theme.
    """

    rcParams.update({
        "figure.facecolor": "#1a1a2e",
        "axes.facecolor": "#16213e",
        "axes.edgecolor": "#444444",
        "axes.labelcolor": "#e0e0e0",
        "text.color": "#e0e0e0",
        "xtick.color": "#aaaaaa",
        "ytick.color": "#aaaaaa",
        "grid.color": "#333333",
        "grid.alpha": 0.4,
        "legend.facecolor": "#1a1a2e",
        "legend.edgecolor": "#444444",
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
    })


_apply_dark_theme()


class Visualization:

    # ---------------------------------------------------------
    # Color Constants
    # ---------------------------------------------------------

    ACCENT_BLUE = "#4fc3f7"
    ACCENT_GREEN = "#81c784"
    ACCENT_RED = "#e57373"
    ACCENT_PURPLE = "#ba68c8"
    ACCENT_ORANGE = "#ffb74d"
    ACCENT_CYAN = "#4dd0e1"

    # ---------------------------------------------------------
    # Display Image
    # ---------------------------------------------------------

    @staticmethod
    def display_image(title, image, cmap=None):
        """
        Display a single image.
        """

        plt.figure(figsize=(6, 6))

        if len(image.shape) == 2:
            plt.imshow(image, cmap=cmap or "gray")
        else:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            plt.imshow(rgb)

        plt.title(title, fontweight="bold", pad=10)
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    # ---------------------------------------------------------
    # Save Comparison Dashboard
    # ---------------------------------------------------------

    @staticmethod
    def save_comparison(images,
                        titles,
                        output_path,
                        columns=3):
        """
        Create a professional comparison dashboard.
        """

        total = len(images)

        rows = (total + columns - 1) // columns

        fig, axes = plt.subplots(
            rows, columns,
            figsize=(5 * columns, 4.5 * rows)
        )

        fig.suptitle(
            "VisionPro Studio — Processing Results",
            fontsize=16,
            fontweight="bold",
            color=Visualization.ACCENT_CYAN,
            y=0.98
        )

        # Flatten axes array
        if rows == 1 and columns == 1:
            axes = np.array([axes])
        axes = np.array(axes).flatten()

        for i, (img, title) in enumerate(zip(images, titles)):

            ax = axes[i]

            if len(img.shape) == 2:
                ax.imshow(img, cmap="gray")
            else:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax.imshow(rgb)

            ax.set_title(
                title,
                fontsize=10,
                fontweight="bold",
                color=Visualization.ACCENT_BLUE,
                pad=8
            )
            ax.axis("off")

            # Subtle border
            for spine in ax.spines.values():
                spine.set_edgecolor("#444444")
                spine.set_linewidth(0.5)

        # Hide unused axes
        for j in range(total, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        Path(output_path).parent.mkdir(
            parents=True,
            exist_ok=True
        )

        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
            facecolor=fig.get_facecolor()
        )

        plt.close()

    # ---------------------------------------------------------
    # Grayscale Histogram
    # ---------------------------------------------------------

    @staticmethod
    def grayscale_histogram(image,
                            output_path=None):
        """
        Plot grayscale histogram with dark theme.
        """

        if len(image.shape) == 3:
            image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY
            )

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.hist(
            image.ravel(),
            bins=256,
            range=(0, 256),
            color=Visualization.ACCENT_CYAN,
            alpha=0.85,
            edgecolor="none"
        )

        ax.set_title(
            "Grayscale Histogram",
            fontweight="bold",
            fontsize=14
        )
        ax.set_xlabel("Pixel Intensity")
        ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.3)

        # Add stats annotation
        mean_val = np.mean(image)
        std_val = np.std(image)

        ax.axvline(
            mean_val, color=Visualization.ACCENT_RED,
            linestyle="--", linewidth=1.5,
            label=f"Mean: {mean_val:.1f}"
        )

        ax.legend(
            loc="upper right",
            fontsize=9,
            framealpha=0.7
        )

        plt.tight_layout()

        if output_path:

            Path(output_path).parent.mkdir(
                parents=True,
                exist_ok=True
            )

            plt.savefig(
                output_path,
                dpi=300,
                bbox_inches="tight",
                facecolor=fig.get_facecolor()
            )

            plt.close()

        else:
            plt.show()

    # ---------------------------------------------------------
    # RGB Histogram
    # ---------------------------------------------------------

    @staticmethod
    def rgb_histogram(image,
                      output_path=None):
        """
        Plot RGB histogram with dark theme.
        """

        colors_map = [
            (Visualization.ACCENT_BLUE, "Blue"),
            (Visualization.ACCENT_GREEN, "Green"),
            (Visualization.ACCENT_RED, "Red"),
        ]

        fig, ax = plt.subplots(figsize=(10, 5))

        for i, (color, label) in enumerate(colors_map):

            hist = cv2.calcHist(
                [image],
                [i],
                None,
                [256],
                [0, 256]
            )

            ax.plot(
                hist,
                color=color,
                alpha=0.8,
                linewidth=1.5,
                label=label
            )

        ax.set_title(
            "RGB Histogram",
            fontweight="bold",
            fontsize=14
        )
        ax.set_xlabel("Pixel Intensity")
        ax.set_ylabel("Frequency")
        ax.set_xlim([0, 256])
        ax.grid(True, alpha=0.3)
        ax.legend(
            loc="upper right",
            fontsize=9,
            framealpha=0.7
        )

        plt.tight_layout()

        if output_path:

            Path(output_path).parent.mkdir(
                parents=True,
                exist_ok=True
            )

            plt.savefig(
                output_path,
                dpi=300,
                bbox_inches="tight",
                facecolor=fig.get_facecolor()
            )

            plt.close()

        else:
            plt.show()

    # ---------------------------------------------------------
    # Before / After Comparison
    # ---------------------------------------------------------

    @staticmethod
    def before_after(original,
                     processed,
                     title1="Original",
                     title2="Processed",
                     output_path=None):
        """
        Compare original and processed image side by side.
        """

        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(14, 6)
        )

        fig.suptitle(
            "Before / After Comparison",
            fontsize=14,
            fontweight="bold",
            color=Visualization.ACCENT_CYAN
        )

        # Original
        if len(original.shape) == 2:
            ax1.imshow(original, cmap="gray")
        else:
            ax1.imshow(
                cv2.cvtColor(
                    original,
                    cv2.COLOR_BGR2RGB
                )
            )

        ax1.set_title(
            title1,
            color=Visualization.ACCENT_GREEN,
            fontweight="bold"
        )
        ax1.axis("off")

        # Processed
        if len(processed.shape) == 2:
            ax2.imshow(processed, cmap="gray")
        else:
            ax2.imshow(
                cv2.cvtColor(
                    processed,
                    cv2.COLOR_BGR2RGB
                )
            )

        ax2.set_title(
            title2,
            color=Visualization.ACCENT_ORANGE,
            fontweight="bold"
        )
        ax2.axis("off")

        plt.tight_layout()

        if output_path:

            Path(output_path).parent.mkdir(
                parents=True,
                exist_ok=True
            )

            plt.savefig(
                output_path,
                dpi=300,
                bbox_inches="tight",
                facecolor=fig.get_facecolor()
            )

            plt.close()

        else:
            plt.show()

    # ---------------------------------------------------------
    # Processing Summary Dashboard
    # ---------------------------------------------------------

    @staticmethod
    def processing_summary(results,
                           output_path):
        """
        Create a large dashboard containing all processed images.
        """

        images = []
        titles = []

        for category in results.values():

            for title, image in category.items():

                titles.append(
                    title.replace("_", " ").title()
                )

                images.append(image)

        Visualization.save_comparison(
            images,
            titles,
            output_path,
            columns=4
        )

    # ---------------------------------------------------------
    # Color Palette Visualization
    # ---------------------------------------------------------

    @staticmethod
    def color_palette(
        colors,
        percentages=None,
        output_path=None
    ):
        """
        Visualize dominant colors as a horizontal palette.

        Parameters
        ----------
        colors : list or numpy.ndarray
            BGR color values.
        percentages : list or numpy.ndarray or None
            Percentage for each color.
        output_path : str or Path or None
            Save path. Shows plot if None.
        """

        n = len(colors)

        if percentages is None:
            percentages = [100 / n] * n

        fig, (ax1, ax2) = plt.subplots(
            2, 1,
            figsize=(12, 3),
            gridspec_kw={"height_ratios": [3, 1]}
        )

        fig.suptitle(
            "Dominant Color Palette",
            fontsize=14,
            fontweight="bold",
            color=Visualization.ACCENT_CYAN
        )

        # Color bars
        start = 0

        for color, pct in zip(colors, percentages):

            width = pct / 100

            # Convert BGR to RGB normalized
            if isinstance(color, np.ndarray):
                rgb = color[::-1] / 255.0
            else:
                rgb = [color[2] / 255, color[1] / 255, color[0] / 255]

            ax1.barh(
                0, width, left=start,
                color=rgb, height=1.0,
                edgecolor="#333333",
                linewidth=0.5
            )

            # Hex label
            hex_color = "#{:02x}{:02x}{:02x}".format(
                int(rgb[0] * 255),
                int(rgb[1] * 255),
                int(rgb[2] * 255)
            )

            if width > 0.08:
                ax1.text(
                    start + width / 2, 0,
                    f"{hex_color}\n{pct:.1f}%",
                    ha="center", va="center",
                    fontsize=8,
                    fontweight="bold",
                    color="white" if sum(rgb) < 1.5 else "black"
                )

            start += width

        ax1.set_xlim(0, 1)
        ax1.set_ylim(-0.5, 0.5)
        ax1.axis("off")

        # Percentage bar chart
        rgbs = []
        for c in colors:
            if isinstance(c, np.ndarray):
                rgbs.append(c[::-1] / 255.0)
            else:
                rgbs.append([c[2] / 255, c[1] / 255, c[0] / 255])

        bars = ax2.bar(
            range(n), percentages,
            color=rgbs,
            edgecolor="#555555",
            linewidth=0.5
        )

        ax2.set_ylabel("%", fontsize=9)
        ax2.set_xticks(range(n))
        ax2.set_xticklabels(
            [f"C{i + 1}" for i in range(n)],
            fontsize=8
        )
        ax2.grid(True, axis="y", alpha=0.3)

        plt.tight_layout()

        if output_path:

            Path(output_path).parent.mkdir(
                parents=True, exist_ok=True
            )

            plt.savefig(
                output_path,
                dpi=300,
                bbox_inches="tight",
                facecolor=fig.get_facecolor()
            )
            plt.close()

        else:
            plt.show()

    # ---------------------------------------------------------
    # Metrics Overlay
    # ---------------------------------------------------------

    @staticmethod
    def metrics_overlay(
        image,
        metrics,
        output_path=None
    ):
        """
        Display image with key metrics overlaid.

        Parameters
        ----------
        image : numpy.ndarray
            Input image.
        metrics : dict
            Key metrics to display.
        output_path : str or Path or None
            Save path.
        """

        fig, (ax1, ax2) = plt.subplots(
            1, 2,
            figsize=(14, 6),
            gridspec_kw={"width_ratios": [2, 1]}
        )

        fig.suptitle(
            "Image Analysis Summary",
            fontsize=14,
            fontweight="bold",
            color=Visualization.ACCENT_CYAN
        )

        # Image
        if len(image.shape) == 2:
            ax1.imshow(image, cmap="gray")
        else:
            ax1.imshow(
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            )

        ax1.set_title("Source Image", fontweight="bold")
        ax1.axis("off")

        # Metrics table
        ax2.axis("off")
        ax2.set_title(
            "Image Metrics",
            fontweight="bold",
            color=Visualization.ACCENT_GREEN
        )

        y_pos = 0.95

        for key, value in metrics.items():

            if isinstance(value, dict):
                continue

            ax2.text(
                0.05, y_pos,
                f"{key}:",
                transform=ax2.transAxes,
                fontsize=10,
                fontweight="bold",
                color=Visualization.ACCENT_BLUE,
                verticalalignment="top"
            )

            ax2.text(
                0.55, y_pos,
                str(value),
                transform=ax2.transAxes,
                fontsize=10,
                color="#e0e0e0",
                verticalalignment="top"
            )

            y_pos -= 0.06

        plt.tight_layout()

        if output_path:

            Path(output_path).parent.mkdir(
                parents=True, exist_ok=True
            )

            plt.savefig(
                output_path,
                dpi=300,
                bbox_inches="tight",
                facecolor=fig.get_facecolor()
            )
            plt.close()

        else:
            plt.show()