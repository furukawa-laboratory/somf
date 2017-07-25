# KSEstandard.pyのメモ
## KSE0331からの変更点
### 変更済み
- $\beta_0$の定義を以下に変更   $\frac{1}{\beta_0}=\frac{1}{GD}\sum_{ij}h_{ij}||y_i-x_j||^2$
- $A$の定義式を以下に変更
$a_{ni}=r_{ni}'\{\beta_n(\varphi_{nk}-\bar{\varphi}_n)+\frac{\beta_nE_n-D}{2(1+g_n)}\}$
- 勾配を求める式を次のように変更
$\frac{\partial F}{\partial z_n}=\sum_i\bar{A}_{ni}\delta_{ni}^*-\alpha z_n$  
$\delta_{ni}^*=\gamma\delta_{ni}$
- 更新式を次のように変更
$z_n:=z_n+\frac{\varepsilon}{\gamma}\frac{\partial F}{\partial z_n}$

### 二度目の変更
- 勾配を次のように変更．
$\frac{\partial F}{\partial z_n}=\sum_i\bar{A}_{ni}\delta_{ni}^*-\alpha D z_n$

- 更新式を次のように変更
$z_n:=z_n+\frac{\varepsilon}{\gamma D}\frac{\partial F}{\partial z_n}$

### これから変更したい点
- $\alpha$と$\gamma$の定義の変更
    - $\gamma=1.0$としていたのを$\alpha=1.0$に変更．fitで$\gamma$を指定する．
    - 事前分布の精度$\alpha=1/\sigma_z^2$，平滑化カーネルの精度$\gamma=1/\sigma_r^2$より，$\sigma_z$と$\sigma_r$の比が一定であれば同じ効果が得られるはず
      - 331で$\gamma=1.0, \alpha=1/30^2$と指定していた場合，$\gamma=30^2, \alpha=1.0$と指定すれば同じことになるはず．
