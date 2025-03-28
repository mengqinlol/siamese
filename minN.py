import heapq

class MinN:
    def __init__(self, N=10):
        self.heap = []           # 使用最大堆维护前N小元素（存储负值实现）
        self.label_counts = {}   # 实时统计各标签出现次数
        self.capacity = N

    def add(self, value, label):
        if len(self.heap) < self.capacity:
            heapq.heappush(self.heap, (-value, label))
            self.label_counts[label] = self.label_counts.get(label, 0) + 1
        else:
            current_max = -self.heap[0][0]
            if value < current_max:
                # 移除当前最大元素，并更新统计
                removed_val, removed_label = heapq.heappop(self.heap)
                self.label_counts[removed_label] -= 1
                if self.label_counts[removed_label] == 0:
                    del self.label_counts[removed_label]
                # 添加新元素
                heapq.heappush(self.heap, (-value, label))
                self.label_counts[label] = self.label_counts.get(label, 0) + 1

    def get(self):
        if not self.label_counts:
            return None  # 处理空的情况
        max_label = max(self.label_counts.items(), key=lambda x: x[1])[0]
        return max_label
    
    def get_sorted_list(self):
        return sorted(self.label_counts.items(), key=lambda x: x[1], reverse=True)

    def get_first_label(self):
        return self.heap[0][1] if self.heap else None