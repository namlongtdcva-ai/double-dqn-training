"""
- Double DQN
  + MLP (256,128,64) 
  + GAMMA=0.1 
  + Huber (smooth L1) 
  + Clip gradient 1.0 
  + Adam optimizer
  + LR = 3e-4 + cosine theo epoch
  + Replay buffer lưu CHỈ SỐ mẫu (state lấy từ tensor train)
  + Epsilon giảm tuyến tính theo bước (ở train.py)
  + Target sao chép mỗi TARGET_UPDATE bước 
  + Xử lý theo batch (vector hoá), không dùng cờ `done`
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from preprocessing import CFG


class QNet(nn.Module):
    def __init__(self, d_in, n_act, hidden):
        super().__init__()
        layers, prev = [], d_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, n_act))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class IndexReplayBuffer:
    """Lưu index/action/reward/next-index; state lấy từ tensor train khi cần."""

    def __init__(self, capacity, device):
        self.capacity, self.pos, self.size, self.device = capacity, 0, 0, device
        self.idx = torch.zeros(capacity, dtype=torch.long, device=device)
        self.act = torch.zeros(capacity, dtype=torch.long, device=device)
        self.rew = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.nxt = torch.zeros(capacity, dtype=torch.long, device=device)

    def push(self, idx, act, rew, nxt):
        ind = (torch.arange(len(idx), device=self.device) + self.pos) % self.capacity
        self.idx[ind], self.act[ind], self.rew[ind], self.nxt[ind] = idx, act, rew, nxt
        self.pos = (self.pos + len(idx)) % self.capacity
        self.size = min(self.size + len(idx), self.capacity)

    def sample(self, b):
        j = torch.randint(0, self.size, (b,), device=self.device)
        return self.idx[j], self.act[j], self.rew[j], self.nxt[j]

class DoubleDQNAgent:
    def __init__(self, state_dim, n_actions, device, cfg=CFG):
        self.cfg, self.device, self.n_actions, self.gamma = cfg, device, n_actions, cfg["GAMMA"]
        self.online = QNet(state_dim, n_actions, cfg["HIDDEN"]).to(device)
        self.target = QNet(state_dim, n_actions, cfg["HIDDEN"]).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.opt = torch.optim.Adam(self.online.parameters(), lr=cfg["LR"])
        self.sched = (torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=cfg["EPOCHS"])
                      if cfg["COSINE_LR"] else None)
        self.buffer = IndexReplayBuffer(cfg["BUFFER_SIZE"], device)

    def act(self, states, eps):
        """Epsilon-greedy theo batch: với xác suất eps chọn ngẫu nhiên, ngược lại argmax Q của mạng online."""
        n = len(states)
        with torch.no_grad():
            greedy = self.online(states).argmax(1)
        rand_a = torch.randint(0, self.n_actions, (n,), device=self.device)
        return torch.where(torch.rand(n, device=self.device) < eps, rand_a, greedy)

    def update(self, X_t):
        """Một lần cập nhật Double DQN từ replay buffer; trả về loss (tensor 0-chiều, đã detach)."""
        b_idx, b_a, b_r, b_nxt = self.buffer.sample(self.cfg["BATCH"])
        q = self.online(X_t[b_idx]).gather(1, b_a[:, None]).squeeze(1)
        with torch.no_grad():
            a_star = self.online(X_t[b_nxt]).argmax(1, keepdim=True)          # chọn action bằng online
            q_next = self.target(X_t[b_nxt]).gather(1, a_star).squeeze(1)     # đánh giá bằng target
            y_t = b_r + self.gamma * q_next
        loss = F.smooth_l1_loss(q, y_t)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 1.0)
        self.opt.step()
        return loss.detach()

    def sync_target(self):
        self.target.load_state_dict(self.online.state_dict())

    def end_epoch(self):
        if self.sched is not None:
            self.sched.step()

    @property
    def lr(self):
        return self.opt.param_groups[0]["lr"]

    @torch.no_grad()
    def q_values(self, Xnp, bs=65536):
        self.online.eval()
        out = [self.online(torch.as_tensor(Xnp[i:i + bs], device=self.device)).float().cpu().numpy()
               for i in range(0, len(Xnp), bs)]
        self.online.train()
        return np.concatenate(out)

    def save(self, path):
        torch.save(self.online.state_dict(), path)

    def load(self, path):
        self.online.load_state_dict(torch.load(path, map_location=self.device))