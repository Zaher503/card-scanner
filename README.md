# card_scanner

Scans cards with a webcam. Hold a card under the camera, and when a rectangle with
text on it stays still for about 0.8 s, it gets cropped, straightened and saved.
Take the card away before showing the next one.

## Setup

```
cd tools/card_scanner
uv sync
```

## Usage

```
uv run python card_scanner.py "scans/Tier 1"
```

- Saves `card_001.jpg`, `card_002.jpg`, ... in the folder. With no folder it uses `scans/<date_time>`.
  If you rerun into the same folder, numbering picks up where it stopped.
- Outline colors: green = card with text, hold still (the bar fills up). Orange = rectangle
  but no text found. Cyan = already saved, remove the card.
- Keys: `space` captures right away, `u` deletes the last capture, `q` or `esc` quits.
- `t` turns auto-capture off and on. Start with `--no-auto` to get focus right on the first
  card (use `space` or just watch sharpness), then press `t` to let it auto-capture the rest.
- `--camera 1` picks another webcam. `--raw` also saves the full uncropped frame.
- `--hold 1.2` waits longer before capturing. `--min-text 0.02` needs more text before it
  counts as a card.
- Focus: autofocus is on by default. `[` and `]` switch to manual focus and step it down/up,
  `a` toggles autofocus, `c` opens the webcam's own settings dialog. The preview shows the
  current focus and a sharpness number, so step focus until sharpness is highest, then start
  with `--focus <that value>` next time.
- Auto-capture saves the sharpest frame from the hold. `--min-sharp 150` skips captures below
  that sharpness (watch the number on screen to pick a value for your camera).
- For best results use a dark, plain background, even light and no glare.

Afterwards, run the OCR tool on the folder (the tier is the first number in the folder name):

```
cd ../card_ocr
uv run python card_ocr.py "../card_scanner/scans/Tier 1"
```
