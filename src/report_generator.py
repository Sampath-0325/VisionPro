"""
==============================================================
VisionPro Studio
Report Generator Module
==============================================================

Generates professional reports for processed images.

Supports:
    • TXT Report
    • Markdown Report
    • HTML Report
    • CSV Metrics Export
    • JSON Report

Author : VisionPro Studio Team
"""

import csv
import json
from pathlib import Path
from datetime import datetime


class ReportGenerator:

    def __init__(self, report_directory="reports"):

        self.report_directory = Path(report_directory)

        self.report_directory.mkdir(
            parents=True,
            exist_ok=True
        )

    # -------------------------------------------------------
    # Internal Helpers
    # -------------------------------------------------------

    @staticmethod
    def _separator(length=70):

        return "=" * length

    @staticmethod
    def _time():

        return datetime.now().strftime(
            "%d-%m-%Y %H:%M:%S"
        )

    @staticmethod
    def _flatten_stats(statistics):
        """Flatten nested statistics dict for tabular output."""

        flat = {}

        for key, value in statistics.items():

            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    flat[f"{key} — {sub_key}"] = sub_value
            else:
                flat[key] = value

        return flat

    # -------------------------------------------------------
    # TXT REPORT
    # -------------------------------------------------------

    def generate_text_report(
        self,
        metadata,
        statistics,
        applied_operations,
        output_files,
        filename="report.txt"
    ):

        report_path = self.report_directory / filename

        flat_stats = self._flatten_stats(statistics)

        with open(report_path, "w", encoding="utf-8") as report:

            report.write(self._separator())
            report.write("\n")
            report.write("VISIONPRO STUDIO")
            report.write("\n")
            report.write("Advanced Image Processing Report")
            report.write("\n")
            report.write(self._separator())
            report.write("\n\n")

            report.write(
                f"Generated : {self._time()}\n\n"
            )

            # --------------------------------------------
            # Image Metadata
            # --------------------------------------------

            report.write("IMAGE INFORMATION\n")
            report.write("-" * 50 + "\n")

            for key, value in metadata.items():

                if key == "EXIF":
                    report.write("\nEXIF Data\n")
                    for ek, ev in value.items():
                        report.write(
                            f"  {ek:<22}: {ev}\n"
                        )
                else:
                    report.write(
                        f"{key:<25}: {value}\n"
                    )

            report.write("\n")

            # --------------------------------------------
            # Statistics
            # --------------------------------------------

            report.write("IMAGE METRICS\n")
            report.write("-" * 50 + "\n")

            for key, value in flat_stats.items():

                report.write(
                    f"{key:<25}: {value}\n"
                )

            report.write("\n")

            # --------------------------------------------
            # Operations
            # --------------------------------------------

            report.write("PROCESSING OPERATIONS\n")
            report.write("-" * 50 + "\n")

            for operation in applied_operations:

                report.write(f"✓ {operation}\n")

            report.write("\n")

            # --------------------------------------------
            # Files
            # --------------------------------------------

            report.write("GENERATED FILES\n")
            report.write("-" * 50 + "\n")

            for file in output_files:

                report.write(f"{file}\n")

            report.write("\n")

            report.write(self._separator())
            report.write("\n")
            report.write("End of Report\n")

        return report_path

    # -------------------------------------------------------
    # MARKDOWN REPORT
    # -------------------------------------------------------

    def generate_markdown_report(
        self,
        metadata,
        statistics,
        applied_operations,
        output_files,
        filename="report.md"
    ):

        report_path = self.report_directory / filename

        flat_stats = self._flatten_stats(statistics)

        with open(report_path, "w", encoding="utf-8") as report:

            report.write("# VisionPro Studio\n\n")

            report.write(
                "## Advanced Image Processing Report\n\n"
            )

            report.write(
                f"**Generated:** {self._time()}\n\n"
            )

            report.write("---\n\n")

            # Metadata

            report.write("## Image Information\n\n")

            report.write("| Property | Value |\n")
            report.write("|:---------|:------|\n")

            for key, value in metadata.items():

                if key == "EXIF":
                    continue

                report.write(
                    f"| {key} | {value} |\n"
                )

            report.write("\n")

            # EXIF if present
            if "EXIF" in metadata:
                report.write("### EXIF Data\n\n")
                report.write("| Tag | Value |\n")
                report.write("|:----|:------|\n")

                for ek, ev in metadata["EXIF"].items():
                    report.write(f"| {ek} | {ev} |\n")

                report.write("\n")

            # Metrics

            report.write("## Image Metrics\n\n")

            report.write("| Metric | Value |\n")
            report.write("|:-------|:------|\n")

            for key, value in flat_stats.items():

                report.write(
                    f"| {key} | {value} |\n"
                )

            report.write("\n")

            # Operations

            report.write("## Operations Performed\n\n")

            for item in applied_operations:

                report.write(f"- ✓ {item}\n")

            report.write("\n")

            # Files

            report.write("## Output Files\n\n")

            for file in output_files:

                report.write(f"- `{file}`\n")

            report.write("\n---\n")
            report.write(
                f"\n*Report generated by VisionPro Studio v2.0*\n"
            )

        return report_path

    # -------------------------------------------------------
    # HTML REPORT
    # -------------------------------------------------------

    def generate_html_report(
        self,
        metadata,
        statistics,
        applied_operations,
        output_files,
        filename="report.html"
    ):
        """
        Generate a styled HTML report.
        """

        report_path = self.report_directory / filename

        flat_stats = self._flatten_stats(statistics)

        html = []

        html.append("<!DOCTYPE html>")
        html.append('<html lang="en">')
        html.append("<head>")
        html.append('<meta charset="UTF-8">')
        html.append('<meta name="viewport" content="width=device-width, initial-scale=1.0">')
        html.append("<title>VisionPro Studio Report</title>")
        html.append("<style>")
        html.append("""
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, sans-serif;
                background: #0f0f23;
                color: #e0e0e0;
                padding: 40px;
                line-height: 1.6;
            }
            .container { max-width: 900px; margin: 0 auto; }
            h1 {
                color: #4dd0e1;
                font-size: 2rem;
                margin-bottom: 5px;
                border-bottom: 2px solid #4dd0e1;
                padding-bottom: 10px;
            }
            .subtitle {
                color: #888;
                font-size: 0.9rem;
                margin-bottom: 30px;
            }
            h2 {
                color: #81c784;
                font-size: 1.3rem;
                margin-top: 30px;
                margin-bottom: 15px;
                padding-left: 10px;
                border-left: 4px solid #81c784;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 20px;
                background: #1a1a2e;
                border-radius: 8px;
                overflow: hidden;
            }
            th {
                background: #16213e;
                color: #4fc3f7;
                padding: 12px 15px;
                text-align: left;
                font-weight: 600;
            }
            td {
                padding: 10px 15px;
                border-bottom: 1px solid #2a2a4a;
            }
            tr:last-child td { border-bottom: none; }
            tr:hover td { background: #1e1e3e; }
            ul { padding-left: 20px; margin-bottom: 20px; }
            li {
                padding: 4px 0;
                list-style: none;
            }
            li::before {
                content: "✓ ";
                color: #81c784;
                font-weight: bold;
            }
            .file-list li::before {
                content: "📄 ";
            }
            .footer {
                margin-top: 40px;
                padding-top: 20px;
                border-top: 1px solid #333;
                color: #666;
                font-size: 0.8rem;
                text-align: center;
            }
        """)
        html.append("</style>")
        html.append("</head>")
        html.append("<body>")
        html.append('<div class="container">')

        # Header
        html.append("<h1>VisionPro Studio</h1>")
        html.append(
            f'<p class="subtitle">Advanced Image Processing Report — {self._time()}</p>'
        )

        # Metadata
        html.append("<h2>Image Information</h2>")
        html.append("<table>")
        html.append("<tr><th>Property</th><th>Value</th></tr>")

        for key, value in metadata.items():
            if key == "EXIF":
                continue
            html.append(
                f"<tr><td>{key}</td><td>{value}</td></tr>"
            )

        html.append("</table>")

        # Metrics
        html.append("<h2>Image Metrics</h2>")
        html.append("<table>")
        html.append("<tr><th>Metric</th><th>Value</th></tr>")

        for key, value in flat_stats.items():
            html.append(
                f"<tr><td>{key}</td><td>{value}</td></tr>"
            )

        html.append("</table>")

        # Operations
        html.append("<h2>Operations Performed</h2>")
        html.append("<ul>")

        for op in applied_operations:
            html.append(f"<li>{op}</li>")

        html.append("</ul>")

        # Files
        html.append("<h2>Generated Files</h2>")
        html.append('<ul class="file-list">')

        for file in output_files:
            html.append(f"<li>{file}</li>")

        html.append("</ul>")

        # Footer
        html.append('<div class="footer">')
        html.append("Generated by VisionPro Studio v2.0")
        html.append("</div>")

        html.append("</div>")
        html.append("</body>")
        html.append("</html>")

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(html))

        return report_path

    # -------------------------------------------------------
    # CSV EXPORT
    # -------------------------------------------------------

    def generate_csv_report(
        self,
        metadata,
        statistics,
        filename="metrics.csv"
    ):
        """
        Export metrics as CSV.
        """

        report_path = self.report_directory / filename

        flat_stats = self._flatten_stats(statistics)

        # Combine metadata + stats
        all_data = {}
        all_data.update(metadata)
        all_data.update(flat_stats)

        # Remove non-serializable
        clean = {}
        for key, value in all_data.items():
            if isinstance(value, dict):
                continue
            clean[key] = value

        with open(report_path, "w", encoding="utf-8", newline="") as f:

            writer = csv.writer(f)
            writer.writerow(["Metric", "Value"])

            for key, value in clean.items():
                writer.writerow([key, value])

        return report_path

    # -------------------------------------------------------
    # JSON EXPORT
    # -------------------------------------------------------

    def generate_json_report(
        self,
        metadata,
        statistics,
        applied_operations,
        output_files,
        filename="report.json"
    ):
        """
        Export full report as JSON.
        """

        report_path = self.report_directory / filename

        flat_stats = self._flatten_stats(statistics)

        report_data = {
            "generator": "VisionPro Studio v2.0",
            "generated_at": self._time(),
            "image_information": {},
            "image_metrics": flat_stats,
            "operations": applied_operations,
            "output_files": output_files,
        }

        # Clean metadata for JSON
        for key, value in metadata.items():
            if isinstance(value, dict):
                report_data["image_information"][key] = value
            else:
                report_data["image_information"][key] = value

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, default=str)

        return report_path

    # -------------------------------------------------------
    # Console Summary
    # -------------------------------------------------------

    @staticmethod
    def print_summary(
        metadata,
        statistics
    ):

        print("\n")

        print("=" * 60)

        print("VISIONPRO STUDIO SUMMARY")

        print("=" * 60)

        print("\nImage Information\n")

        for key, value in metadata.items():

            if key == "EXIF":
                print(f"\n{'EXIF Data':<22}")
                for ek, ev in value.items():
                    print(f"  {ek:<20}: {ev}")
            else:
                print(f"{key:<22}: {value}")

        print("\nImage Metrics\n")

        for key, value in statistics.items():

            if isinstance(value, dict):
                print(f"\n  {key}:")
                for sk, sv in value.items():
                    print(f"    {sk:<20}: {sv}")
            else:
                print(f"{key:<22}: {value}")

        print("\n" + "=" * 60)

    # -------------------------------------------------------
    # Build Output File List
    # -------------------------------------------------------

    @staticmethod
    def build_file_list(results):

        files = []

        for folder, images in results.items():

            for filename in images.keys():

                files.append(
                    f"outputs/{folder}/{filename}.jpg"
                )

        return files