# 何のリポジトリ？
古川研究室でラボメンバーが共通して用いるライブラリを共有するためのリポジトリ．

# ディレクトリ構成

- libs：外側から参照するライブラリ（packageもしくはmodule）のコードを置くディレクトリ
    - datasets：データセットをまとめたpackage
    - models：SOMなどの学習アルゴリズムのクラスを収めたmoduleをまとめたpackage
    - visualization：描画用package
- tests：libs/modelsにおいたmoduleのペアプロコードを置くディレクトリ
- tutorials：ライブラリのチュートリアルを置くディレクトリ

# 使い方

## ライブラリを使うとき
gitのsubmodule機能を活用して，各自のプロジェクトにflibを置いてください．

参考：Qiita Git submoduleの基礎  
https://qiita.com/sotarok/items/0d525e568a6088f6f6bb

## 機能を追加するとき
ブランチモデルを現在検討中…