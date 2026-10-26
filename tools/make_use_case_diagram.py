"""Draw assets/use_case_diagram.png with matplotlib (UML-style use case diagram)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle

fig, ax = plt.subplots(figsize=(13, 8.8))
ax.set_xlim(0, 13); ax.set_ylim(-0.5, 8.5); ax.axis("off")

ax.add_patch(Rectangle((2.7, 0.0), 7.6, 8.2, fill=False, lw=1.5))
ax.text(6.5, 7.9, "Student Service Request Monitoring System", ha="center", fontweight="bold", fontsize=12)

LX, RX = 4.4, 8.4
uc = {  # name: (x, y)
    "Submit Service Request": (LX, 7.2),
    "Track Request Status": (LX, 6.4),
    "Process Service Request": (LX, 5.6),
    "Update Request Status": (LX, 4.8),
    "Compute Delay Risk Score": (LX, 3.6),
    "Review Delay Risk": (RX, 4.4),
    "Record Final Decision": (RX, 3.45),
    "View SLA Performance": (RX, 2.5),
    "Export Performance Report": (RX, 1.55),
    "Upload & Validate Dataset": (RX, 0.6),
}
for name, (x, y) in uc.items():
    ax.add_patch(Ellipse((x, y), 2.9, 0.65, fc="white", ec="black"))
    ax.text(x, y, name, ha="center", va="center", fontsize=8.5)


def actor(x, y, label):
    ax.add_patch(plt.Circle((x, y + 0.55), 0.18, fc="white", ec="black"))
    ax.plot([x, x], [y + 0.37, y - 0.2], "k"); ax.plot([x - 0.3, x + 0.3], [y + 0.2, y + 0.2], "k")
    ax.plot([x, x - 0.25], [y - 0.2, y - 0.7], "k"); ax.plot([x, x + 0.25], [y - 0.2, y - 0.7], "k")
    ax.text(x, y - 1.0, label, ha="center", va="top", fontsize=8.5)


REQ, STAFF, ADMIN = (1.2, 5.8), (11.8, 5.4), (11.8, 2.0)
actor(*REQ, "<<actor>>\nRequester\n(Student / Faculty / Staff)")
actor(*STAFF, "<<actor>>\nOffice Staff")
actor(*ADMIN, "<<actor>>\nAdministrator")


def line(a, b):
    ax.plot([a[0], b[0]], [a[1], b[1]], color="black", lw=0.8)


for n in ["Submit Service Request", "Track Request Status", "Process Service Request", "Update Request Status"]:
    line((REQ[0] + 0.3, REQ[1] + 0.2), (uc[n][0] - 1.45, uc[n][1]))
for n in ["Process Service Request", "Update Request Status"]:
    line((STAFF[0] - 0.3, STAFF[1] + 0.2), (uc[n][0] + 1.45, uc[n][1]))
for n in ["Review Delay Risk", "Record Final Decision"]:
    line((STAFF[0] - 0.3, STAFF[1] + 0.2), (uc[n][0] + 1.45, uc[n][1]))
for n in ["Review Delay Risk", "Record Final Decision", "View SLA Performance",
          "Export Performance Report", "Upload & Validate Dataset"]:
    line((ADMIN[0] - 0.3, ADMIN[1] + 0.2), (uc[n][0] + 1.45, uc[n][1]))


def dashed(a, b, label, dx=0.12, dy=0.0, ha="left"):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", ls="--", lw=1.0))
    ax.text((a[0] + b[0]) / 2 + dx, (a[1] + b[1]) / 2 + dy, label, ha=ha, va="center", fontsize=8, style="italic")


# <<include>>: Review Delay Risk -> Compute Delay Risk Score
dashed((RX - 1.25, 4.25), (LX + 1.25, 3.75), "<<include>>", dx=0, dy=0.2, ha="center")
# <<extend>>: extending case -> base case
dashed((RX, 3.45 + 0.325), (RX, 4.4 - 0.325), "<<extend>>")
dashed((RX, 1.55 + 0.325), (RX, 2.5 - 0.325), "<<extend>>")

ax.text(6.5, -0.3, "Updated use case diagram - new: Record Final Decision, Upload & Validate Dataset, Administrator actor",
        ha="center", fontsize=8.5)
out = Path(__file__).resolve().parents[1] / "assets" / "use_case_diagram.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
