CustomerSegmentation

CustomerSegmentation เป็นโปรเจกต์ตัวอย่างสำหรับ Customer Segmentation โดยใช้ Python และ scikit-learn เพื่อทำ unsupervised learning แบ่งกลุ่มลูกค้าตาม Age, Annual Income, และ Spending Score พร้อมการวิเคราะห์และ visualization

🔹 Features

ใช้ StandardScaler เพื่อ standardize ข้อมูล

PCA สำหรับแสดงผล 2D visualization ของ clusters

Clustering methods:

KMeans – หาจำนวน cluster ที่เหมาะสมด้วย Silhouette Score

Agglomerative Clustering – Hierarchical clustering

DBSCAN – Density-based clustering

Evaluation metrics:

Silhouette Score

Calinski-Harabasz Index

Davies-Bouldin Index

Visualization ของ cluster ใน 2D PCA space
