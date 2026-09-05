"""Turn the executed notebook into a printable PDF handout."""
import subprocess, sys, pathlib, re

NB   = "regression_models.ipynb"
HTML = pathlib.Path("/tmp/handout.html")
PDF  = pathlib.Path("regression_models_handout.pdf")

PRINT_CSS = """
<style>
@page { size: A4; margin: 15mm 13mm 16mm 13mm; }

html, body { background: #ffffff !important; }
body { font-size: 10.5pt; }

/* strip JupyterLab chrome that means nothing on paper */
.jp-Notebook { padding: 0 !important; }
.jp-Cell { padding: 0 !important; margin: 0 0 8px 0 !important; }
.jp-Cell-inputWrapper, .jp-Cell-outputWrapper { margin: 0 !important; }
.jp-InputPrompt, .jp-OutputPrompt { display: none !important; }
.jp-InputArea-editor { border: none !important; }

/* code blocks: light card, monospace, wrap rather than clip */
.jp-CodeCell .jp-InputArea-editor,
.jp-CodeCell .jp-Cell-inputWrapper .highlight, .jp-CodeCell .jp-Cell-inputWrapper pre {
  background: #f7f7f5 !important;
  border-left: 3px solid #2a78d6 !important;
  border-radius: 3px;
}
.jp-InputArea-editor pre, .jp-OutputArea-output pre {
  font-size: 9pt !important; line-height: 1.42 !important;
  white-space: pre-wrap !important; word-break: break-word;
}
.jp-OutputArea-output pre { background: #fbfbfa !important; color: #1c1c1a !important; }

/* prose */
.jp-RenderedHTMLCommon { font-size: 10.5pt; line-height: 1.55; color: #17171a; }
.jp-RenderedHTMLCommon p { margin: 0.45em 0; }
.jp-RenderedHTMLCommon h1 { font-size: 20pt; margin: 18px 0 8px; color: #10233d;
                            border-bottom: 2px solid #2a78d6; padding-bottom: 5px; }
.jp-RenderedHTMLCommon h2 { font-size: 14pt; margin: 14px 0 6px; color: #10233d; }
.jp-RenderedHTMLCommon h3 { font-size: 11.5pt; margin: 12px 0 4px; color: #1c5cab; }
.jp-RenderedHTMLCommon blockquote {
  border-left: 3px solid #eb6834; background: #fdf6f2;
  padding: 6px 12px; margin: 10px 0; font-style: normal;
}
.jp-RenderedHTMLCommon table { font-size: 9pt; border-collapse: collapse; }
.jp-RenderedHTMLCommon th { background: #eef4fc !important; color: #10233d; }
.jp-RenderedHTMLCommon td, .jp-RenderedHTMLCommon th {
  border: 1px solid #dcdcd8; padding: 3px 7px;
}
.jp-RenderedHTMLCommon hr { border: none; border-top: 1px solid #dcdcd8; margin: 16px 0; }

/* tables produced by pandas */
table.dataframe { font-size: 8.5pt !important; }
table.dataframe th { background: #eef4fc !important; }

img, .jp-RenderedImage img { max-width: 100% !important; height: auto !important; }

/* pagination */
.jp-OutputArea-child, .jp-Cell-inputWrapper, table, pre, img { break-inside: avoid; page-break-inside: avoid; }
h1, h2, h3 { break-after: avoid; page-break-after: avoid; }
</style>
"""


def main():
    print("1/3  notebook -> html")
    subprocess.run(
        [sys.executable, "-m", "jupyter", "nbconvert", "--to", "html",
         "--template", "lab", "--HTMLExporter.theme=light",
         "--output", str(HTML), NB],
        check=True, capture_output=True,
    )

    print("2/3  injecting print stylesheet")
    html = HTML.read_text(encoding="utf-8")
    html = html.replace("</head>", PRINT_CSS + "</head>", 1)
    HTML.write_text(html, encoding="utf-8")

    print("3/3  html -> pdf (chromium)")
    chromium = "/opt/pw-browsers/chromium/chrome-linux/chrome"
    if not pathlib.Path(chromium).exists():
        chromium = subprocess.run(["bash", "-lc", "ls /opt/pw-browsers/chromium*/chrome-linux*/chrome | head -1"],
                                  capture_output=True, text=True).stdout.strip()
    subprocess.run(
        [chromium, "--headless=new", "--disable-gpu", "--no-sandbox",
         "--virtual-time-budget=30000", "--run-all-compositor-stages-before-draw",
         "--no-pdf-header-footer", f"--print-to-pdf={PDF.resolve()}",
         HTML.resolve().as_uri()],
        check=True, capture_output=True,
    )
    print(f"wrote {PDF}  ({PDF.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
