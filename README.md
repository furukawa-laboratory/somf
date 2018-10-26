# What is this?
This is Repository to share library which all members use.  
古川研メンバーが共通して用いるようなライブラリを共有するためのリポジトリ．  


# Directories

```
.
├── libs                    # libraries ライブラリ
│   ├── datasets            # module to return datasets データセットを返すモジュール
│   ├── models              # class of models such as SOM, TSOM... SOMやTSOMなどの学習アルゴリズムのクラス
│   ├── tools               # utility tools あらゆるライブラリで共通して用いるような汎用的なもの
│   └── visualization       # module to visualize such as U-matrix, Component Plane... 描画用のライブラリ
├── playground              # prototype code with playing 試しに作ってみた系のコード
├── tests                   # code to test libraries 開発した学習アルゴリズムが正しく動作するかどうかチェックするための実行ファイル置き場．
│                           # place test code when you review pull request プルリクエストのレビューの際の検証コードはここに置く
└── tutorials               # tutorial of libraries アルゴリズムのチュートリアルを実行するファイル置き場
```

# How to use?

## How to introduce into your repository リポジトリへの導入方法
Please introduce flib into your repository by using **submodule** which is git function. search "submodule git" on google!  
gitの**submodule**機能を用いて自分のリポジトリにflibを導入してください．


※submoduleは簡単に言うとリポジトリの中にリポジトリを入れる機能です．これを使うと以下のようなことができます
- 自分の研究プロジェクトのリポジトリ内にflibを入れることができる（自分のリポジトリからflibにあるライブラリを利用できる）
- リモートのflibでプッシュがあるとそれをフェッチしてマージできる（もちろんそれをしないことも可能）

ネット上ではコマンドラインからの導入の解説がほどんどですが，ある程度source treeで管理することもできます．

詳しくは以下をチェックするか検索してみてください．  
[Git submodule の基礎 - Qiita](https://qiita.com/sotarok/items/0d525e568a6088f6f6bb)  
[Git submoduleの押さえておきたい理解ポイントのまとめ - Qiita](https://qiita.com/kinpira/items/3309eb2e5a9a422199e9)

### submoduleで導入した後の設定（Pycharm）
Pycharm上で利用する場合は，flibのディレクトリをSource rootに設定すると，flibのチュートリアルコードと同様の書き方でライブラリを利用することができます．ただし，大元のリポジトリの中にflib内のディレクトリと同じ名前のディレクトリがあるするとコンフリクトを起こしたりするので注意してください．

## How to request bug fix and new feature? バグ修正や新機能追加の依頼の出し方
Please create **Issues**, use template.  
Issueを作成してください．テンプレートがあるので基本的にはそれを利用してください．

## How to edit exiting code and add new code コードの編集や追加の方法
GitHub-flowと呼ばれる方法を採用します．以下の流れです．
1. Confirm related Issue. If there is no issue, create the issue.  対応するIssueの内容を確認する．ない場合はまずIssueを作成する
2. Create a branch. ブランチを切る．
   - name issue number at prefix, compact representation about the work. ブランチ名はIssue番号を先頭につけて，そのあと作業内容を簡潔に書く．
3. Push at an appropriate frequency. 適宜pushをする．
4. Completing the work, create pull request. 作業がひと段落したらpull requestを出す．
   - Don't forget to write issue number. Issueの番号を必ず書いてください．
   - must request review to someone. レビューを必ず他の人に依頼すること．
      - Doctor students can merge only. Dの学生だけがマージが可能．

GitHub-flowに関してはこちらを参照してください．  
[GitHub Flow ～GitHubを活用するブランチモデル～](https://tracpath.com/bootcamp/learning_git_github_flow.html)


# Note 諸注意
このリポジトリにあるライブラリはあくまでラボメンバー皆で作って行くものです．バグのない完成したものではありません．なのでエンドユーザー感覚で完全にブラックボックスとして使わないでください．masterにマージされているライブラリでも，デバッグをやるぐらいの心づもりで利用してください．また，欲しい機能は自分で作ったりIssueで提案したり，積極的なcontributeを期待しています！
