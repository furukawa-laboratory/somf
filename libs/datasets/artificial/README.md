# sin_two_inputs
以下の関数に従う人工データを生成する．

<img src="https://latex.codecogs.com/gif.latex?g(z_1,z_2)=\sin{\left\{2\pi&space;f(z_1&plus;z_2)\right\}}" title="g(z_1,z_2)=\sin{\left\{2\pi f(z_1+z_2)\right\}}" />

**parameters**
- nb_samples1, nb_samples2: int
   - 各ドメインのオブジェクト数
- frequency: float
   - 周波数<img src="https://latex.codecogs.com/gif.latex?f" title="f" />
   - デフォルトは3/8（岩崎さんの博論の(4.4)式に準拠）
- observed_dim: int
   - 出力される2者の関係データの観測データの次元数
   - デフォルトは1
- retz: bool
   - 真の潜在変数を返すかどうか指定するbool値
   - デフォルトはFalse
