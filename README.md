# Three-Echelon Supply Chain RL Environment

A custom Gymnasium environment simulating a three-echelon supply chain (Farm → Factory → Retail) for reinforcement learning experiments using Stable-Baselines3.

## 📚 Dependencies

This project relies on the following Python libraries:

- `gymnasium` – for the custom environment framework  
- `numpy` – for numerical operations  
- `matplotlib` – for visualization and plotting (optional)  
- `stable-baselines3` – for reinforcement learning algorithms (PPO, DQN, etc.)

Install all dependencies with:

## Limitations
-Lead times between echelons (e.g., farm to factory, factory to retail) are not modeled.

-Any customer demand that cannot be fulfilled on the same day is not carried over to the following day. Instead, it is treated as lost sales.

## ⚠️ Disclaimer / 免責事項

本プロジェクトのコードは教育目的・研究目的で提供されています。  
このコードの使用により発生した **損害・損失・不具合・誤動作** について、作者は一切の責任を負いません。

**Use at your own risk.**  
The author is not responsible for any direct or indirect damages resulting from the use of this software.


```bash
pip install -r requirements.txt




