"""
- Môi trường học tăng cường dạng bảng cho bài toán phân loại tấn công V2X (6 lớp).
- Các thành phần:
  state      : vector đặc trưng đã chuẩn hoá của một dòng log
  action     : nhãn dự đoán 0..K-1
  reward     : +w[y] nếu đúng, -w[y] nếu sai;  w_k = sqrt(n_max / n_k), chuẩn hoá để trung bình = 1
  next state : mẫu kế tiếp trong một hoán vị ngẫu nhiên của tập train (các mẫu độc lập nhau)

- Hai lớp:
  WeightedTabularRLEnv : API gymnasium từng bước (dùng để kiểm thử, minh hoạ)
  TorchBatchEnv        : bản vector hoá trên GPU, dùng trong train.py
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from preprocessing import K


def class_weights_from_labels(y, num_classes=K):
    cnt = np.bincount(y, minlength=num_classes).astype(np.float64)
    w_raw = np.sqrt(cnt.max() / np.maximum(cnt, 1.0))
    return (w_raw / w_raw.mean()).astype(np.float32)


class WeightedTabularRLEnv(gym.Env):
    """Env gymnasium từng bước: nhận mảng X (N, D) đã chuẩn hoá và nhãn y (N,) thuộc 0..K-1."""

    def __init__(self, X, y, num_classes=K, shuffle=True, class_weights=None, seed=None):
        super().__init__()
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int64)
        assert len(self.X) == len(self.y) and len(self.X) > 1
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.num_samples = len(self.y)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.X.shape[1],), dtype=np.float32)
        self.action_space = spaces.Discrete(num_classes)
        self.class_weights = (class_weights_from_labels(self.y, num_classes) if class_weights is None
                              else np.asarray(class_weights, dtype=np.float32))
        self.rng = np.random.default_rng(seed)
        self.indices = np.arange(self.num_samples)
        self.current_step = 0

    @property
    def current_label(self):
        """Nhãn thật của mẫu hiện tại (dùng khi đánh giá)."""
        return int(self.y[self.indices[self.current_step]])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if self.shuffle:
            self.rng.shuffle(self.indices)
        self.current_step = 0
        return self.X[self.indices[0]], {}

    def step(self, action):
        if self.current_step >= self.num_samples:
            raise RuntimeError("Episode đã kết thúc, hãy gọi reset().")
        target = self.y[self.indices[self.current_step]]
        w = float(self.class_weights[target])
        reward = w if int(action) == int(target) else -w        

        self.current_step += 1
        done = self.current_step >= self.num_samples
        if not done:
            next_state = self.X[self.indices[self.current_step]]
        else:
            next_state = np.zeros(self.observation_space.shape, dtype=np.float32)
        return next_state, reward, done, False, {}


class TorchBatchEnv:
    """Bản vector hoá của WeightedTabularRLEnv trên GPU/CPU"""

    def __init__(self, X, y, class_weights, device, batch_size):
        import torch
        self._torch = torch
        self.device = device
        self.X = torch.as_tensor(X, device=device)
        self.y = torch.as_tensor(y, dtype=torch.long, device=device)
        self.w = torch.as_tensor(class_weights, dtype=torch.float32, device=device)
        self.N, self.B = len(self.y), batch_size
        self.steps_per_epoch = (self.N - 1) // self.B        
        self.perm = None

    def reset(self):
        """Bắt đầu epoch mới: hoán vị ngẫu nhiên toàn bộ tập train."""
        self.perm = self._torch.randperm(self.N, device=self.device)

    def batch(self, step):
        """Trả (idx, nidx): chỉ số các state hiện tại và state kế tiếp (mẫu kế trong hoán vị)."""
        B = self.B
        return (self.perm[step * B:(step + 1) * B],
                self.perm[step * B + 1:(step + 1) * B + 1])

    def reward(self, idx, actions):
        yt = self.y[idx]
        return (2.0 * (actions == yt).float() - 1.0) * self.w[yt]