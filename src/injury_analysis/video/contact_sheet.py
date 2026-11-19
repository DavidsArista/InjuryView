from __future__ import annotations

import math
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from injury_analysis.errors import InjuryAnalysisError
from injury_analysis.video.frames import ContactSheetRecord, FrameManifest, FrameRecord


def build_contact_sheets(
    manifest: FrameManifest,
    output_dir: Path,
    *,
    columns: int = 3,
    frames_per_sheet: int = 6,
    directory: str = "contact_sheets",
    sheet_prefix: str = "sheet",
) -> FrameManifest:
    if columns < 1 or frames_per_sheet < 1:
        raise InjuryAnalysisError("Contact sheet layout must use positive sizes")
    sheets_dir = output_dir / directory
    if sheets_dir.exists():
        shutil.rmtree(sheets_dir)
    sheets_dir.mkdir(parents=True, exist_ok=True)

    records: list[ContactSheetRecord] = []
    chunks = [
        manifest.frames[index : index + frames_per_sheet]
        for index in range(0, len(manifest.frames), frames_per_sheet)
    ]
    for number, chunk in enumerate(chunks, start=1):
        sheet_id = f"{sheet_prefix}_{number:03d}"
        filename = f"{sheet_id}.jpg"
        _render_sheet(chunk, sheets_dir / filename, output_dir, sheet_id=sheet_id, columns=columns)
        records.append(
            ContactSheetRecord(
                sheet_id=sheet_id,
                path=f"{directory}/{filename}",
                frame_ids=[frame.frame_id for frame in chunk],
            )
        )
    return manifest.model_copy(
        update={
            "contact_sheets": records,
            "sheet_columns": columns,
            "frames_per_sheet": frames_per_sheet,
        }
    )


def _render_sheet(
    frames: list[FrameRecord],
    destination: Path,
    output_dir: Path,
    *,
    sheet_id: str,
    columns: int,
) -> None:
    thumb_width = 360
    label_height = 36
    header_height = 32
    padding = 8
    opened: list[tuple[FrameRecord, Image.Image]] = []
    for frame in frames:
        path = output_dir / frame.path
        if not path.is_file():
            raise InjuryAnalysisError(f"Extracted frame is missing: {frame.path}")
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            ratio = thumb_width / rgb.width
            thumb_height = max(1, int(round(rgb.height * ratio)))
            opened.append(
                (frame, rgb.resize((thumb_width, thumb_height), Image.Resampling.LANCZOS))
            )
    cell_height = label_height + max(image.height for _, image in opened)
    rows = math.ceil(len(opened) / columns)
    width = padding + columns * (thumb_width + padding)
    height = header_height + padding + rows * (cell_height + padding)
    canvas = Image.new("RGB", (width, height), (20, 24, 28))
    draw = ImageDraw.Draw(canvas)
    font = _font(18)
    header_font = _font(16)
    draw.text((padding, 6), sheet_id, fill=(230, 230, 230), font=header_font)

    for index, (frame, thumb) in enumerate(opened):
        column = index % columns
        row = index // columns
        x = padding + column * (thumb_width + padding)
        y = header_height + padding + row * (cell_height + padding)
        draw.rectangle((x, y, x + thumb_width, y + label_height), fill=(32, 36, 42))
        label = f"{frame.frame_id}  {frame.time_s:.2f}s"
        if frame.clip_id:
            label = f"{frame.frame_id}  {frame.clip_id}  {frame.time_s:.2f}s"
        draw.text((x + 8, y + 8), label, fill=(245, 245, 245), font=font)
        image_y = y + label_height + max(0, (cell_height - label_height - thumb.height) // 2)
        canvas.paste(thumb, (x, image_y))
    canvas.save(destination, format="JPEG", quality=85)


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()
