from pathlib import Path

src = Path("calibrate_reliability.py")
dst = Path("calibrate_new_model.py")

text = src.read_text(encoding="utf-8")

text = text.replace(
    r"models\tumorguard_dev.pth",
    r"models\tumorguard_model.pt"
)

text = text.replace(
    "006 to 020",
    "041 to 053"
)

text = text.replace(
    "range(6, 21)",
    "range(41, 54)"
)

dst.write_text(text, encoding="utf-8")

print("Created:", dst)
