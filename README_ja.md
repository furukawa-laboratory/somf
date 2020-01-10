# What is this?
自己組織化マップ(Self-Organizing Map: SOM)とその派生アルゴリズムのPythonによる実装を公開しているリポジトリです。  

本リポジトリは九州工業大学の古川研究室が運営を行なっています。本研究室はSOMをベースとした、データからの知識発見に役立つ機械学習アルゴリズムを開発しています。本リポジトリではベーシックなSOMに限らず、本研究室で開発したアルゴリズムの実装も公開しています。

本リポジトリは元々、研究室内でアルゴリズムを共有するためにprivateで整備されていたものですが、この度publicとしました。現在ライブラリとしての公開を目指して整備を行なっています。もちろん外部の方からのcontributionは大歓迎です。

# SOMとは？
SOMはニューラルネットワークの一種で、高次元データに対する可視化やモデリングに幅広く用いられているアルゴリズムです。古くから存在するアルゴリズムではありますが、近年盛り上がりを見せる機械学習・深層学習のアルゴリズムにも通じる側面があります。詳細は本研究室で公開している[ドキュメント](http://www.brain.kyutech.ac.jp/~furukawa/data/SOMtext.pdf)をご覧ください。


# 現在公開しているコード

## アルゴリズム
- SOM（バッチ型）
   - tensorflow ver
   - numpy ver
- [CCA-SOM](https://www.jstage.jst.go.jp/article/jsoft/30/2/30_525/_article/-char/ja)（マルチビューデータに対応したSOM）
- [テンソルSOM](https://www.sciencedirect.com/science/article/pii/S0893608016000149)（テンソルデータに対応したSOM）
   - tensorflow ver
   - numpy ver
- Unsupervised Kernel Regression
- Kernel smoothing(Nadaraya-Watson estimater)

## 可視化ツール
- Grad_norm for SOM
- Conditional Component Plane for TSOM

## データセット
- [Beverage Preference Data set](http://www.brain.kyutech.ac.jp/~furukawa/beverage-e/)
- 各種人工データセット

# User guide
現在整備中です。手っ取り早く動かしたい方は[tutorials](https://github.com/furukawa-laboratory/somf/tree/master/tutorials)に実行コードがありますのでそちらをお試しください。
