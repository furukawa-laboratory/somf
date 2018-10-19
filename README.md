# What is this? 何のリポジトリ？
古川研メンバーが共通して用いるようなライブラリを共有するためのリポジトリ．
This is Repository to share library which all members use.

# Directories ディレクトリ構成

```
.
├── libs                    # ライブラリの置き場
│   ├── datasets            # データセットを返す関数
│   ├── models              # SOMなどの学習アルゴリズムのクラス
│   ├── tools               # あらゆるライブラリで共通して用いるような汎用的なもの
│   └── visualization       # 描画用のライブラリ
├── tests                   # 開発した学習アルゴリズムが正しく動作するかどうかチェックするための実行ファイル置き場．
│                           # プルリクエストのレビューの際の検証コードはここに置く
└── tutorials               # アルゴリズムのチュートリアルを実行するファイル置き場
```

# How to use? 使い方

## ライブラリをユーザーとして使うとき
gitの**submodule**機能を用いて自分のリポジトリにflibを導入してください．
Please introduce flib into your repository by using **submodule** which is git function.

submoduleは簡単に言うとリポジトリの中にリポジトリを入れる機能です．これを使うと以下のようなことができます
- 自分の研究プロジェクトのリポジトリ内にflibを入れることができる（自分のリポジトリからflibにあるライブラリを利用できる）
- リモートのflibでプッシュがあるとそれをフェッチしてマージできる（もちろんそれをしないことも可能）

ネット上ではコマンドラインからの導入の解説がほどんどですが，ある程度source treeで管理することもできます．

詳しくは以下をチェックしてください．  
[Git submodule の基礎 - Qiita](https://qiita.com/sotarok/items/0d525e568a6088f6f6bb)  
[Git submoduleの押さえておきたい理解ポイントのまとめ - Qiita](https://qiita.com/kinpira/items/3309eb2e5a9a422199e9)

## 機能を追加するとき
編集中…
