
import matplotlib.pyplot as plt

counts = [20, 40, 60, 80, 100]
no_sharing = [131.75, 244.46, 306.14, 437.78, 511.66]
kmeans = [93.8, 131.03, 183.23, 209.14, 237.11]

plt.figure(figsize=(8, 5))

plt.plot(counts, no_sharing, marker='o', markersize=8, linewidth=2, label='Independent')
plt.plot(counts, kmeans, marker='s', markersize=8, linewidth=2, label='Opportunistic D2D')

plt.xlim(10, 110)
plt.ylim(0, 800)

plt.xlabel('Number of Target points', fontsize=12)
plt.ylabel('Time (s)', fontsize=12)
plt.title('Simulation Time by Number of Target points', fontsize=14)
plt.xticks(counts)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=11)
plt.tight_layout()
plt.show()