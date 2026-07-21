# Stereophonic Studio

音源をアップロードせず、**ブラウザの中だけ**で AI（Demucs）が楽器ごとに分離し、
分離した各パートを 3D 空間に配置して再生できる静的Webアプリです。

- 分離はすべて利用者のブラウザ内で実行（サーバー不要・音源は外部に出ない）
- 静的ホスティング（GitHub Pages 等）に置くだけで、誰でもURLから利用可能

## 使い方
`index.html` を開く → 「分離してプレイヤーを開く」→ 曲を選ぶ → 分離 → 立体音響で再生。

## 構成
- `index.html` … ライブラリ／入口
- `player.html` … 立体音響プレイヤー＋ブラウザ内分離パネル
- `vendor/demucs-web/` … 分離エンジン（timcsy/demucs-web, MIT）
- `coi-serviceworker.js` … 静的ホストでSharedArrayBufferを有効化（gzuidhof, MIT）

分離モデル（約172MB, ONNX）は Hugging Face Hub から読み込みます。

推奨ブラウザ: Chrome / Edge（WebGPU）。
