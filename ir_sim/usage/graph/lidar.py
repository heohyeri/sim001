import matplotlib.pyplot as plt

lidar_ranges = [8, 12, 16, 20, 24]
no_sharing = [293.37, 306.14, 299.75, 309.92, 301.57]
kmeans = [184.8, 183.23, 135.18, 129.57, 133.79]

plt.figure(figsize=(8, 5))

plt.plot(lidar_ranges, no_sharing, marker='o', markersize=8, linewidth=2, label='No Sharing')
plt.plot(lidar_ranges, kmeans, marker='s', markersize=8, linewidth=2, label='K-Means')

plt.xlim(6, 26)
plt.ylim(50, 400)

plt.xlabel('LiDAR Range (m)', fontsize=12)
plt.ylabel('Time (s)', fontsize=12)
plt.title('Simulation Time by LiDAR Range', fontsize=14)
plt.xticks(lidar_ranges)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=11)
plt.tight_layout()
plt.show()
