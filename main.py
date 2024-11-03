import numpy as np
import cv2

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import torch
import torch.nn as nn
from torchvision import transforms

from torch.utils.data import DataLoader
from torchvision.models import resnet34, ResNet34_Weights

from data_utils import Data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def extract_feature(dataloader, model):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for data in dataloader:
            img, label = data
            img = img.to(device)
            feature = model(img)
            features.append(feature.cpu().numpy())
            labels.append(label.numpy())
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    return features.squeeze(), labels

def evaluation(label, pred):
    acc = accuracy_score(label, pred)
    f1 = f1_score(label, pred, average='macro')
    precision = precision_score(label, pred, average='macro')
    recall = recall_score(label, pred, average='macro')
    return acc, f1, precision, recall

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((224, 224))])  # ResNet 的輸入尺寸
dataset = Data('./Mini', train_n=50, test_n=10, transform=transform)
train_data, test_data = dataset.train_data, dataset.test_data
train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=10)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=10)

cnn_model = resnet34(weights=ResNet34_Weights.DEFAULT)
cnn_model = nn.Sequential(*list(cnn_model.children())[:-1]).to(device)

train_X, train_label = extract_feature(train_loader, cnn_model)
test_feature, test_label = extract_feature(test_loader, cnn_model)
print(train_X.shape, train_label.shape)
print(test_feature.shape, test_label.shape)

knn = KNeighborsClassifier(n_neighbors=11, n_jobs=-1)
knn.fit(train_X, train_label)
knn_prob = knn.predict_proba(test_feature)
knn_pred = np.argmax(knn_prob, axis=1)
test_acc, test_f1, test_prec, test_recall = evaluation(test_label, knn_pred)
print(f'KNN: {test_acc:.4f}, {test_f1:.4f}, {test_prec:.4f}, {test_recall:.4f}')


svm = SVC(probability=True)
svm.fit(train_X, train_label)
svm_prob = svm.predict_proba(test_feature)
svm_pred = np.argmax(svm_prob, axis=1)
test_acc, test_f1, test_prec, test_recall = evaluation(test_label, svm_pred)
print(f'SVM: {test_acc:.4f}, {test_f1:.4f}, {test_prec:.4f}, {test_recall:.4f}')


rf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
rf.fit(train_X, train_label)
rf_prob = rf.predict_proba(test_feature)
rf_pred = np.argmax(rf_prob, axis=1)
test_acc, test_f1, test_prec, test_recall = evaluation(test_label, rf_pred)
print(f'RF: {test_acc:.4f}, {test_f1:.4f}, {test_prec:.4f}, {test_recall:.4f}')


# mix_test_prob = 0.6*svm_prob + 0.4*knn_prob
# mix_test_prob = 0.5*svm_prob + 0.5*knn_prob
mix_test_prob = 0.5*svm_prob + 0.5*rf_prob

pred_test_class = np.argmax(mix_test_prob, axis=1)
test_acc, test_f1, test_prec, test_recall = evaluation(test_label, pred_test_class)
print(f'Mix: {test_acc:.4f}, {test_f1:.4f}, {test_prec:.4f}, {test_recall:.4f}')
