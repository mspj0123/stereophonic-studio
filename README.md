# Stereophonic Studio（ブラウザ内 音源分離＋立体音響プレイヤー）

公開URL: https://mspj0123.github.io/stereophonic-studio/

曲を**アップロードせず**、あなたのブラウザの中だけで AI（Demucs）が
vocals / drums / bass / other に分離し、そのまま3D空間に配置して再生します。
音源が外部サーバーに送られることはありません。

## 使い方

1. ページを開く
2. 「📂 作業フォルダを選ぶ」で保存先を決める（初回だけ・PCのChrome / Edge）
3. 「＋ 分離する曲を選ぶ」で曲を選ぶ → 分離（数十秒〜数分）
4. 分離結果は `<作業フォルダ>/<曲名>/` に自動保存され、次回は上の「曲:」から選ぶだけで再生できる
5. 「▶ 再生」→「立体音響」ON で3D再生。パッドで各パートの位置を動かせる

Firefox / Safari / iPhone では作業フォルダへの直接保存に対応していないため、
分離後に出る**ダウンロードボタン**で保存し、再生時は「📁 フォルダから読み込む」を使ってください。

## ファイル構成

- `index.html` … 本体（3Dプレイヤー＋分離パネル＋作業フォルダ）
- `player.html` … 旧URL用の転送ページ（`index.html` へリダイレクト）
- `separator-worker.js` … 分離処理を行う Web Worker
- `vendor/demucs-web/` … 分離エンジン（MIT）
- `coi-serviceworker.js` … 静的ホストで SharedArrayBuffer を有効化する（MIT）
- `songs/` + `library.json` … 作者が用意する公開ライブラリ（任意）
- `docs/` … 設計書・テスト設計・テスト結果

## 注意

- 推奨: PC の Chrome / Edge。初回のみ分離モデル（約172MB）を読み込みます（以後は端末内キャッシュ）
- 分離はメモリを使うため、対応は10分まで（メモリの少ない端末は5分まで）
