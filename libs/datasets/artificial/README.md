# sin_two_inputs
以下の関数に従う人工データを生成する．

<img src="https://latex.codecogs.com/gif.latex?g(z_1,z_2)=\sin{\left\{2\pi&space;f(z_1&plus;z_2)\right\}}" title="g(z_1,z_2)=\sin{\left\{2\pi f(z_1+z_2)\right\}}" />

ただし<img src="https://latex.codecogs.com/gif.latex?z_1,z_2\in[-1,&plus;1]" title="z_1,z_2\in[-1,+1]" />で乱数により生成する．

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
   
   
**returns**
- x: ndarray
   - shape=(nb_samples1, nb_samples2, observed_dim)
   
   
# animal
出典元のデータを古川研の教育用に改竄した古川研オリジナル動物データ

## 出典
T. Kohonen, [Self-Organizing Maps], Springer Series in Information Sciences, Vol.30(1995)
