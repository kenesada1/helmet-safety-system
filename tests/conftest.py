from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))


def make_voc_xml(
    *,
    filename: str = "sample.jpg",
    width: int = 100,
    height: int = 200,
    depth: int = 3,
    objects: tuple[tuple[str, int, int, int, int], ...] = (),
    folder: str = "JPEGImages",
    path: str = "C:/dataset/sample.jpg",
) -> str:
    """生成小型 VOC XML 字符串，供单元测试按需组合尺寸、类别和检测框。"""

    object_xml = "".join(
        f"""
        <object>
          <name>{name}</name>
          <bndbox>
            <xmin>{xmin}</xmin><ymin>{ymin}</ymin>
            <xmax>{xmax}</xmax><ymax>{ymax}</ymax>
          </bndbox>
        </object>
        """
        for name, xmin, ymin, xmax, ymax in objects
    )
    return f"""<annotation>
      <folder>{folder}</folder>
      <filename>{filename}</filename>
      <path>{path}</path>
      <size><width>{width}</width><height>{height}</height><depth>{depth}</depth></size>
      {object_xml}
    </annotation>"""
