# pai-final-language-pick-place

Physical AI 応用1 最終課題。言葉で指示すると、Franka Panda が箱を並べ替えるシミュレーション(Genesis)。

「一番大きい赤い箱を左の皿に」「小さい青い箱を緑の箱の右隣に」のような、大きさ・色・位置関係を含む指示を、VLM(Gemini)が画像から解釈する。

## 仕組み
1. 真上のカメラ画像と指示文を Gemini に渡し、「掴む箱」と「置く場所」を画像上の点で返してもらう(`points`)。
2. 点を机上の座標に変換し、IK + 関節補間で上から掴んで運ぶ(`ppl/sim.py`)。
3. 止まらないための保険:
   - VLM の点がどの箱にも当たらない、または出力が解釈できない → 各物体に番号を描いた画像で聞き直す(`marks`)。
   - 物理的に掴めない → 箱をハンドに追従させる方式に切り替える(今回の実行では使われていない)。
   - Gemini が混雑・制限中(503/429) → 再試行し、次のモデルへ自動で切り替える。
4. 成功した計画は `plan_cache.json` に保存される。再実行では API を使わずに同じ動作を再生できる(`--no-cache` で無効化)。

## 実行方法 (Colab)
1. `notebooks/colab_demo.ipynb` を Colab で開き、ランタイムを GPU にする。
2. 左の鍵アイコン(シークレット)に `GEMINI_API_KEY` を登録し、ノートブックからのアクセスをオンにする。キーはコードやリポジトリに含めない。
3. 上から順に実行する。最初のセルは GitHub からクローンする(Google ドライブのフォルダで作業するなら `USE_DRIVE = True`)。
4. API なしでシミュレータだけ確認するなら `python run.py --mock "mock test" --out mock.mp4`。

## ローカルでの実行
```bash
pip install -r requirements.txt
export GEMINI_API_KEY=...          # Windows PowerShell: $env:GEMINI_API_KEY="..."
python run.py "一番大きい赤い箱を、左の皿に置いて" "小さい青い箱を、緑の箱の右隣に置いて" --out demo.mp4
pytest                              # 座標変換・Gemini応答の解析・モデル切り替えのテスト (API不要)
```
`--fps` で動画の速さ、`--font` で字幕に使う日本語フォントを指定できる。Colab では追加の設定なしで動く。

## 構成
- `ppl/geometry.py` 画像 ↔ 机上座標の変換、位置関係(左隣・右隣など)
- `ppl/vlm.py` Gemini 呼び出し、モデル切り替え、応答の解析
- `ppl/sim.py` シーン(箱6個・皿2枚)、ロボット制御、録画
- `run.py` 実行エントリ(計画 → 実行 → 動画保存)
- `make_final_video.py` 公式の OP / ED とデモ動画をつなぐ(OP / ED はリポジトリに含まれない)
- `plan_cache.json` デモ実行で Gemini が返した計画の記録
- `docs/report.html` 提出レポートの元(PDF はここから作成)
- `debug_render.py` 描画が動いているかの確認用

## 動作環境の確認状況
Genesis 1.4.5。Colab の T4 GPU でデモ(5本の指示)を実行して動画を作成した。ローカルの Windows では CPU でモック実行のみ確認した。
