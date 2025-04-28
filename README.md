# LunarLander-v3 Benchmark — DQN Family vs PPO

This repository contains:

| Folder | Purpose |
|--------|---------|
| `notebooks/` | **`lunar_lander_benchmark.ipynb`** – executed notebook with code, plots, and milestone videos/GIFs. |
| `videos/`    | Raw MP4 clips saved during milestone evaluation. |
| `gifs/`      | Lightweight GIF previews generated from the MP4s (render inline on GitHub). |
| `plots/`     | All result charts saved as PNGs (also embedded inline in the notebook). |
| `models/`    | Final trained model checkpoints (DQN `.pth`, PPO `.zip`). |

> **Tip:** Open the notebook in **Google Colab** or **nbviewer** for full inline video playback.

---

## 🔧  Setup

```bash
git clone https://github.com/your-username/rl-carracing-dqn-vs-ppo.git
cd rl-carracing-dqn-vs-ppo
pip install -r requirements.txt

Python ≥ 3.9, CUDA optional.
Key packages: stable-baselines3, gymnasium[box2d], moviepy, matplotlib, pandas.

⸻

▶️  Quick start (Colab)
	1.	Open notebooks/lunar_lander_benchmark.ipynb in Colab.
	2.	Runtime → Run all (≈ 15 min on a free GPU).
	3.	Scroll to:
	•	Section 2 – Training logs
	•	Section 3 – Result grids (plots)
	•	Section 4 – 5×5 milestone video/GIF gallery

All plots/GIFs are regenerated automatically; MP4 clips are produced every
MILESTONES = [100,200,300,400,500] episodes for seed 0.

⸻

🏗️  Notebook structure

Section	Code cells	What it does
1  Imports & globals	Cell 0	Sets hyper-params, seeds, color palette.
2  Training	Cell 13	train_and_snapshot() trains Vanilla DQN, Double DQN, Dueling DQN, PER-DQN, PPO; logs per-episode CSVs, saves milestone videos and model checkpoints.
3  Post-training plots	Cell 18	Reads CSVs → writes PNGs + displays them inline; covers learning curves, milestone trends, overlays.
4  GIF conversion	Cell 18b	Converts each milestone MP4 to a small GIF (moviepy).
5  Video/GIF gallery	Cell 19	Builds 5 × 5 HTML table (rows = milestones, cols = algos); embeds GIF if present, else MP4 (≤ 4 MB base-64, otherwise thumbnail).
6  Discussion	Cell 20	Brief conclusions + next-step suggestions.



⸻

🔄  Reproduce your own run

python scripts/run_benchmark.py --env LunarLander-v3 \
                                --episodes 500 \
                                --seeds 0 1 2

The script is a thin wrapper around the same functions used in the notebook; it writes logs, plots, and videos to the same folders.

⸻

📊  Evaluation metrics
	•	Reward (mean ± std) per episode and milestone
	•	Success rate (reward ≥ 200) per milestone
	•	Sample efficiency = steps to first success
	•	Training stability = rolling σ₅₀(reward)
	•	Compute cost = mean wall-clock seconds per algorithm

See plots/ or the “Result grids” section in the notebook for visuals.

⸻

❓  FAQ
	•	Why don’t MP4s autoplay on GitHub?
GitHub’s viewer blocks inline MP4. GIF previews are provided; open the notebook in Colab/nbviewer for full playback.
	•	How do I change milestones?
Edit MILESTONES at the top of the notebook and rerun Cells 13, 18, 18b, 19.

⸻

© 2025 NC — MIT License

---

## 📝 2. Inline code-doc suggestions

Add these **doc-strings** at the top of each major function / cell (only one example shown; copy the pattern).

```python
# Cell 13  ──────────────────────────────────────────────────────────────
def train_and_snapshot(name: str, agent_ctor, is_ppo: bool):
    """
    Train an RL agent and log everything needed for the final report.

    Parameters
    ----------
    name : str
        Human-readable algorithm label (e.g. "Vanilla_DQN").
    agent_ctor : Callable
        Factory that returns an *untrained* agent instance.
    is_ppo : bool
        Flag because SB3-PPO API differences (VecEnv, .learn, .predict).

    Side-effects
    ------------
    • Appends per-episode rows to `master_full_training.csv`
    • Appends milestone rows to `master_milestone_metrics.csv`
    • Saves model checkpoints under <BASE_LOG_DIR>/<name>/
    • Records a single MP4 per milestone & seed
    • Returns the trained agent instance
    """

At the top of Cell 18:

# Cell 18  — Post-Training Evaluation & Plotting
"""
Reads the two master CSV logs and generates all required plots:

Row 1 – learning curves  
Row 2 – milestone metrics  
Row 3 – distribution overlays  
Row 4 – compute-cost bar

Each figure is saved to plots/ and also displayed inline via plt.show().
"""

At the top of Cell 18b:

# Cell 18b — MP4 → GIF batch conversion
"""
Converts every milestone MP4 (seed 0) to a small GIF for GitHub-friendly
inline animation. Skips clips that already have a .gif twin.

Requires `moviepy>=1.0`. Typical output size: 2-4 MB per GIF.
"""

At the top of Cell 19:

# Cell 19 — 5 × 5 Milestone Video/GIF Gallery
"""
Builds an HTML table where each row = milestone (100-500), each column =
algorithm. For each cell:

1. If a GIF exists → <img> embed (animates everywhere).  
2. Else if MP4 ≤ 4 MB → base-64 <video> embed (plays inline).  
3. Else            → external <video> tag (click-to-play on GitHub).

Displayed via `IPython.display.HTML`; no disk writes.
"""



