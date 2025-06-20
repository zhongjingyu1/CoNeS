import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans

class PML_Confidence(nn.Module):
    def __init__(self, train_givenY, num_classes, mu=0.99, gamma=0.99):
        super().__init__()
        self.mu = mu
        self.gamma = gamma
        print('Calculating uniform targets...')
        # calculate confidence
        self.confidence = train_givenY.float()/train_givenY.sum(dim=1, keepdim=True)
        self.num_classes = num_classes
        label_correlation_init = np.zeros((num_classes, num_classes))
        for i in range(num_classes):
            indices = np.where(train_givenY.cpu().numpy()[:,i] == 1)
            for j in [x for x in range(num_classes) if x != i]:
                label_correlation_init[i,j]=sum(train_givenY[np.array(indices).ravel(),j])/np.array(indices).ravel().size
        label_correlation_init_norm = (label_correlation_init / label_correlation_init.sum(1))
        np.fill_diagonal(label_correlation_init_norm, 1)
        self.label_correlation = label_correlation_init_norm


    def forward(self, logits, batch_index, targets, r, beta):
        loss_func = F.binary_cross_entropy_with_logits
        epsilon = 1e-8
        prediction_sigmoid = torch.sigmoid(logits)
        xi = 1 / (self.num_classes - targets.sum(1))

        negative = ((1 / (r * -1 * torch.log(1-prediction_sigmoid) + epsilon)) * (1 - targets)) ** (1 / (r - 1))
        negative_sum = ((((1 / (r * -1 * torch.log(1 - prediction_sigmoid) + epsilon)) * (1 - targets)).sum(1)) ** (1 / (r - 1))).unsqueeze(1)

        phi = negative * xi.view(-1, 1) / (negative_sum + epsilon)
        phi_all = (phi+targets).detach()
        loss_bce = loss_func(logits, targets)

        targets_new = (self.confidence[batch_index, :] > (1 / (targets.sum(1)+epsilon)).unsqueeze(1)) * 1
        average_loss = loss_func(logits, targets_new.to(torch.float32), weight=phi_all, reduction="mean")

        return loss_bce + beta * average_loss, loss_bce, average_loss


    @torch.no_grad()
    def confidence_move_update(self, temp_un_conf, partial_label, batch_index):
        prediction_sigmoid = torch.sigmoid(temp_un_conf)
        prediction_adj = (torch.from_numpy(np.dot(prediction_sigmoid.cpu().numpy(), self.label_correlation)).cuda() * partial_label).to(torch.float32)
        prediction_adj_nor = prediction_adj / prediction_adj.sum(dim=1, keepdim=True)
        self.confidence[batch_index, :] = self.mu * self.confidence[batch_index, :] + prediction_adj_nor * (1 - self.mu)
        return None

    @torch.no_grad()
    # 使用原型的k-means
    def correlation_move_update(self, feature_memory, epoch, num_class, kmeans):
        if epoch == 1:
            self.label_correlation = self.label_correlation
        else:
            for k in range(num_class):
                memory_k = feature_memory[k]
                kmeans.fit(memory_k)
                centroids_k = kmeans.cluster_centers_
                for j in [x for x in range(num_class) if x != k]:
                    memory_j = feature_memory[j]
                    kmeans.fit(memory_j)
                    centroids_j = kmeans.cluster_centers_
                    simi = 0
                    for nu in range(kmeans.n_clusters):
                        cos_sim = (F.cosine_similarity(torch.from_numpy(centroids_k[nu]), torch.from_numpy(centroids_j), dim=1)).sum()/kmeans.n_clusters
                        simi = simi + cos_sim
                    simi_mean = simi / kmeans.n_clusters
                    self.label_correlation[k][j] = self.gamma * self.label_correlation[k][j] + simi_mean.numpy() * (1 - self.gamma)
            np.fill_diagonal(self.label_correlation, 0)
            self.label_correlation = (self.label_correlation / self.label_correlation.sum(1))
            np.fill_diagonal(self.label_correlation, 1)
        return None