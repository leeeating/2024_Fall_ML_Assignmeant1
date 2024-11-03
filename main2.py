import numpy as np
import cv2
from scipy.stats import kurtosis, skew

from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import torch
import torch.nn as nn
from torchvision import transforms

from torch.utils.data import DataLoader
from torchvision.models import resnet34, ResNet34_Weights

from data_utils import Data, ImbalancedData

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
    f1 = f1_score(label, pred, average='macro', zero_division=0)
    precision = precision_score(label, pred, average='macro', zero_division=0)
    recall = recall_score(label, pred, average='macro', zero_division=0)
    return acc, f1, precision, recall

def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((224, 224))])  # ResNet 的輸入尺寸
dataset = ImbalancedData('./Mini', train_n=300, test_n=100, transform=transform)
# dataset = Data('./Mini', train_n=50, test_n=10, transform=transform)

train_data, test_data = dataset.train_data, dataset.test_data
train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=10)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=10)

torch.backends.cudnn.benchmark = True
cnn_model = resnet34(weights=ResNet34_Weights.DEFAULT)
cnn_model = nn.Sequential(*list(cnn_model.children())[:-1]).to(device)

train_feature, train_label = extract_feature(train_loader, cnn_model)
test_feature, test_label = extract_feature(test_loader, cnn_model)
print(train_feature.shape, train_label.shape)

kmean = KMeans(n_clusters=100)
kmean.fit(train_feature)

test_kmean_features = kmean.transform(test_feature)
print(test_kmean_features.shape)
test_pred = np.argmin(test_kmean_features, axis=1)
test_dist = np.min(test_kmean_features, axis=1)
print(test_dist.shape)

test_std = np.zeros((len(test_feature)))
for i in range(100):
    in_class_idx = np.where(test_pred==i)
    in_class_mean = np.mean(test_dist[in_class_idx], axis=0)
    in_class_std = np.std(test_dist[in_class_idx], axis=0)
    if in_class_std == 0:
        print(f'Class {i} has std = 0')

    test_std[in_class_idx] = np.abs( test_dist[in_class_idx] - in_class_mean) / in_class_std


# svm = SVC(probability=True, C=1.0, kernel='rbf')
svm_norm = SVC(probability=True, C=1, kernel='rbf')
svm_imba = SVC(probability=True, C=2, kernel='rbf')
svm_norm.fit(train_feature, train_label)
svm_imba.fit(train_feature, train_label)

svm_norm_prob = svm_norm.predict_proba(test_feature)
svm_imba_prob = svm_imba.predict_proba(test_feature)
svm_norm_pred = np.argmax(svm_norm_prob, axis=1)
svm_imba_pred = np.argmax(svm_imba_prob, axis=1)

for std in [1, 1.5, 2]:
    print((test_std>=std).sum())
    hard_idx = np.where(test_std>=std)
    norm_idx = np.where(test_std<std)

    svm_pred = np.zeros_like(test_label)
    svm_pred[hard_idx] = svm_imba_pred[hard_idx]
    svm_pred[norm_idx] = svm_norm_pred[norm_idx]

    acc, f1, precision, recall = evaluation(test_label, svm_norm_pred)
    print(f'SVM Norm: Acc: {acc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}')

    acc, f1, precision, recall = evaluation(test_label, svm_imba_pred)
    print(f'SVM Imba: Acc: {acc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}')

    acc, f1, precision, recall = evaluation(test_label, svm_pred)
    print(f'SVM: Acc: {acc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}')