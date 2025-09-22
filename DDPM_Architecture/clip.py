import os
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import open_clip
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

# Load CLIP model
model_name = "ViT-H-14"
clip_model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained='laion2b_s32b_b79k')
tokenizer = open_clip.get_tokenizer(model_name)

for name, param in clip_model.named_parameters():
    if param.requires_grad:
        print(f"Training: {name}")

# Dataset
class ImageTextMultiTaskDataset(Dataset):
    def __init__(self, image_paths, notes, feature_labels, locus_labels):
        self.image_paths = image_paths
        self.notes = notes
        self.feature_labels = feature_labels
        self.locus_labels = locus_labels
        self.transform = preprocess

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = self.transform(Image.open(self.image_paths[idx]).convert("RGB"))
        text = self.notes[idx]
        feat_label = self.feature_labels[idx]
        locus_label = self.locus_labels[idx]
        return image, text, feat_label, locus_label

# Define classification model
class CLIPMultiTaskClassifier(nn.Module):
    def __init__(self, clip_model, num_feature_classes, num_locus_classes):
        super().__init__()
        self.clip = clip_model

        self.img_dim = clip_model.visual.output_dim
        self.txt_dim = clip_model.text_projection.shape[1]

        self.fusion = nn.Sequential(
            nn.Linear(self.img_dim + self.txt_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.fc_feature = nn.Linear(512, num_feature_classes)
        self.fc_locus = nn.Linear(512, num_locus_classes)

    def forward(self, image, tokenized_text):
        image_features = self.clip.encode_image(image)
        text_features = self.clip.encode_text(tokenized_text)
        combined = torch.cat([image_features, text_features], dim=1)  # [B, img_dim + txt_dim]
        fused = self.fusion(combined)
        feature_logits = self.fc_feature(fused)
        locus_logits = self.fc_locus(fused)
        return feature_logits, locus_logits

# Data preparation

df = pd.read_csv("clean_data2.csv")
image_paths = []    # List of image file paths
folder_path = Path("C:/Users/sarah/OneDrive/Documents/research_data/VU_Text_Sample_512_sarah/images0")
text_notes = []        # List of text notes
locus = []            # List of integer class labels
features = []
# Process each row
for idx, row in df.iterrows():
    image_id = str(row['id'])
    image_id = image_id.strip()
    locus_field = str(row['locus'])
    feature_field = "NULL"
    note_field = "NULL"

    if str(row['notes']) == "" or str(row['feature_type_name']) == "":
        continue
    note_field = str(row['notes'])
    feature_field = str(row['feature_type_name'])

    
    image_name = "locus_" + image_id + ".jpg"
    temp = folder_path / image_name
    image_paths.append(temp)
    text_notes.append(note_field)
    locus.append(locus_field)
    features.append(feature_field)





# Encode feature types
feature_encoder = LabelEncoder()
feature_labels = feature_encoder.fit_transform(features)

# Encode locus types
locus_encoder = LabelEncoder()
locus_labels = locus_encoder.fit_transform(locus)

dataset = ImageTextMultiTaskDataset(image_paths, text_notes, feature_labels, locus_labels)
dataloader = DataLoader(dataset, batch_size=5, shuffle=True)

# Model and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_feature_classes = len(feature_encoder.classes_)
num_locus_classes = len(locus_encoder.classes_)

model = CLIPMultiTaskClassifier(clip_model, num_feature_classes, num_locus_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    list(model.fc_feature.parameters()) + list(model.fc_locus.parameters()),
    lr=1e-4)

# Training loop
for epoch in range(3):
    for images, texts, feat_targets, locus_targets in dataloader:
        images = images.to(device)
        tokenized_texts = tokenizer(texts).to(device)
        feat_targets = feat_targets.to(device)
        locus_targets = locus_targets.to(device)

        optimizer.zero_grad()

        feat_logits, locus_logits = model(images, tokenized_texts)
        feat_loss = criterion(feat_logits, feat_targets)
        locus_loss = criterion(locus_logits, locus_targets)
        loss = feat_loss

        loss.backward()
        optimizer.step()

        print("Loss:", loss.item())
        print("Predicted:", torch.argmax(feat_logits, dim=1))
        print("True:", feat_targets)
