import os
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import label, center_of_mass

# -----------------------------
# 1. Load image
# -----------------------------
img = Image.open(
    '/Users/trumanparrish/Downloads/Sample Generated YY Matrix No Scatter(gb_size = 7, std = 8).png'
).convert("L")


# -----------------------------
# 2. Crop only the actual plot area
# Adjust these numbers if needed
# -----------------------------
img_crop = img.crop((115, 271, 1537, 1693))
#img_crop = img.crop((114, 280, 1537, 1703))
img_big = img_crop.resize((4096,4096))

img_big.save('output2.png') #make active if want to observe the cropped and resized image

gray = np.array(img_big)
plot = gray[0:4096, 0:4096]

height, width = plot.shape

# -----------------------------
# 3. Energy axis range
# -----------------------------
x_min, x_max = 0, 4096
y_min, y_max = 0, 4096

# -----------------------------
# 4. Detect bright coincidence dots
# -----------------------------
threshold = 180

bright_mask = plot >= threshold

labeled, num_features = label(bright_mask)

centers = center_of_mass(
    bright_mask,
    labeled,
    range(1, num_features + 1)
)

coincidences = []

for center in centers:
    y_pixel, x_pixel = center

    energy_x = x_min + (x_pixel / width) * (x_max - x_min)

    # y-axis is flipped in image coordinates
    energy_y = y_max - (y_pixel / height) * (y_max - y_min)

    coincidences.append({
        "Energy_X_keV": round(energy_x / 100) * 100,
        "Energy_Y_keV": round(energy_y / 100) * 100
    })
# -----------------------------
# 4.5. Create Directories
# -----------------------------

out_dir = "ML - Coincidence" #change to avoid overwrite
os.makedirs(out_dir, exist_ok=True)
dir_names = ["Detected Coincidences", "Energy Sees List"]
for j in dir_names:
    path = os.path.join(out_dir, j)
    os.makedirs(path, exist_ok=True)
detected_coinc = os.path.join(out_dir, "Detected Coincidences")
energy_sees = os.path.join(out_dir, "Energy Sees List")

# -----------------------------
# 5. Create coincidence table
# -----------------------------
df = pd.DataFrame(coincidences)

df = df.drop_duplicates()
df = df.sort_values(["Energy_X_keV", "Energy_Y_keV"])

print("Detected Coincidences:")
print(df)

df.to_csv(
    os.path.join(detected_coinc, f"detected_coincidences.csv"),
    index=False
)

# -----------------------------
# 6. Group by energy: what each energy sees
# -----------------------------
sees_list = (
    df.groupby("Energy_X_keV")["Energy_Y_keV"]
    .apply(lambda values: sorted(set(values)))
    .reset_index()
)

sees_list = sees_list.rename(columns={
    "Energy_X_keV": "Energy_keV",
    "Energy_Y_keV": "Sees_keV"
})

print("\nWhat each energy sees:")
print(sees_list)

sees_list.to_csv(
    os.path.join(energy_sees, f"energy_sees_list.csv"),
    index=False
)