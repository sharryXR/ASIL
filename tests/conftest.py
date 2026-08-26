import zipfile
from pathlib import Path

import pytest


@pytest.fixture
def tmp_dir(tmp_path):
    """Provide a temporary directory cleaned up after the test."""
    return tmp_path


@pytest.fixture
def sample_svg(tmp_dir: Path) -> Path:
    """Create a minimal SVG file for testing."""
    svg = tmp_dir / "test.svg"
    svg.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<svg xmlns="http://www.w3.org/2000/svg" '
        'xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" '
        'width="800" height="600" viewBox="0 0 800 600">\n'
        '  <g inkscape:groupmode="layer" inkscape:label="Layer 1" id="layer1">\n'
        '    <rect id="rect1" x="10" y="20" width="100" height="50" '
        'style="fill:#ff0000;stroke:#000000;stroke-width:1"/>\n'
        '    <circle id="circle1" cx="200" cy="150" r="40" '
        'style="fill:#00ff00"/>\n'
        '    <text id="text1" x="300" y="100" '
        'style="font-size:16px;fill:#000000">Hello</text>\n'
        '  </g>\n'
        '  <g inkscape:groupmode="layer" inkscape:label="Layer 2" id="layer2">\n'
        '    <rect id="rect2" x="400" y="300" width="80" height="60" '
        'style="fill:#0000ff"/>\n'
        '  </g>\n'
        '</svg>\n'
    )
    return svg


@pytest.fixture
def sample_ods(tmp_dir: Path) -> Path:
    """Create a minimal ODS file (ZIP with content.xml) for testing."""
    ods = tmp_dir / "test.ods"
    content_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<office:document-content '
        'xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
        'xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0" '
        'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0">\n'
        '  <office:body>\n'
        '    <office:spreadsheet>\n'
        '      <table:table table:name="Sheet1">\n'
        '        <table:table-row>\n'
        '          <table:table-cell office:value-type="string">\n'
        '            <text:p>Revenue</text:p>\n'
        '          </table:table-cell>\n'
        '          <table:table-cell office:value-type="float" office:value="100">\n'
        '            <text:p>100</text:p>\n'
        '          </table:table-cell>\n'
        '        </table:table-row>\n'
        '        <table:table-row>\n'
        '          <table:table-cell office:value-type="string">\n'
        '            <text:p>Cost</text:p>\n'
        '          </table:table-cell>\n'
        '          <table:table-cell office:value-type="float" office:value="60">\n'
        '            <text:p>60</text:p>\n'
        '          </table:table-cell>\n'
        '        </table:table-row>\n'
        '      </table:table>\n'
        '    </office:spreadsheet>\n'
        '  </office:body>\n'
        '</office:document-content>\n'
    )
    with zipfile.ZipFile(ods, "w") as zf:
        zf.writestr("content.xml", content_xml)
        zf.writestr(
            "META-INF/manifest.xml",
            '<?xml version="1.0"?><manifest:manifest xmlns:manifest="urn:oasis:names:tc:opendocument:xmlns:manifest:1.0"/>'
        )
    return ods
