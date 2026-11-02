
import os

DATASET_DIR = r"C:\Users\DELL\Downloads\archive"

train_dir = os.path.join(DATASET_DIR, "train")
test_dir = os.path.join(DATASET_DIR, "test")

print("Checking dataset...\n")

for split, folder in [("TRAIN", train_dir), ("TEST", test_dir)]:
    print(f"\n{split} DATASET")

    if not os.path.exists(folder):
        print("Folder not found!")
        continue

    for emotion in sorted(os.listdir(folder)):
        emotion_path = os.path.join(folder, emotion)

        if os.path.isdir(emotion_path):
            count = len([
                f for f in os.listdir(emotion_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ])
            print(f"{emotion}: {count} images")

print("\nDataset check complete!")
