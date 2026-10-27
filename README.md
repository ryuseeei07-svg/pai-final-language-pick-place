# pai-final-language-pick-place

Physical AI 応用1 最終課題。言葉で指示すると、Franka Panda が箱を片付けるシミュレーション(Genesis)。

## 仕組み
1. 真上のカメラ画像と指示文を Gemini (VLM) に渡し、「掴む箱」と「置く場所」を画像上の点で返してもらう。
2. 点を世界座標に変換し、IK + 関節補間で上から掴んで運ぶ(`ppl/sim.py`)。
3. 失敗時の保険:
   - VLM の点がどの箱にも当たらない → 各物体に番号を描いた画像で再質問(set-of-marks)。
   - 物理的に掴めない → 箱をハンドに追従させる方式に切り替え(動画のキャプションと結果に明記)。

「一番大きい赤い箱」「青の右隣」のような空間的な指示や、複数ステップの指示に対応する。

## 使い方
`notebooks/colab_demo.ipynb` を Colab (GPU) で開く。Gemini の API キーは Colab のシークレット `GEMINI_API_KEY` に登録する(コードやリポジトリには含めない)。

```bash
pip install -r requirements.txt
GEMINI_API_KEY=... python run.py "put the largest red box on the left plate" --out demo.mp4
python run.py --mock "smoke test"   # API なしでシミュレータのみ確認
pytest                              # 幾何変換とVLM応答の解析テスト
```

## 構成
- `ppl/geometry.py` 画像↔世界座標の変換
- `ppl/vlm.py` Gemini 呼び出しと応答の解析
- `ppl/sim.py` シーン、ロボット制御、録画
- `run.py` 実行エントリ
