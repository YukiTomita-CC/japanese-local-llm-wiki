---
displayed_sidebar: handson
description: LoRAを使った教師ありファインチューニング
---

# SFT with LoRA(Decoder Only)
:::warning
この記事は作成途中です！ \
内容の精査が足りていない箇所や誤った情報が含まれている可能性がありますので、参考にしないでください！
:::

ここでは、基盤モデルに対してSFT(教師ありファインチューニング)を行うことで下流タスクに適応させる方法を体験しましょう。  
必要となるVRAMを削減するためのLoRAを使った学習を行います。

## 目的(、成果物、ゴール)
基盤モデルを`(入力)指示 + 要約する文章 -> (出力)要約`の形式に対応させる。

## Hands-on
### 1. 基盤モデルの選定
まずはファインチューニングする基盤モデルを選びましょう。

今現在著名なモデルシリーズにはLlama2、Mistral、Qwenなどがありますが、今回はその中でもライセンス的に制限の少ないMistralを用いることにします。\
Mistralを継続事前学習し日本語に対応させた基盤モデルのうち、企業や大学が公開しているものは下記の3つです。
- [Japanese Stable LM Base Gamma 7B(Stability AI)](https://huggingface.co/stabilityai/japanese-stablelm-base-gamma-7b)
- [Swallow-MS-7b-v0.1(TokyoTech-LLM)](https://huggingface.co/tokyotech-llm/Swallow-MS-7b-v0.1)
- [RakutenAI-7B(Rakuten)](https://huggingface.co/Rakuten/RakutenAI-7B)

では今回の要約タスクに適していると思われるモデルはどれでしょうか？🤔

幸いなことに[RakutenAI-7Bのテクニカルレポート](https://arxiv.org/pdf/2403.15484#page=4)や[Swallow-MS-7b-v0.1のモデルカード](https://huggingface.co/tokyotech-llm/Swallow-MS-7b-v0.1#japanese-tasks)に他モデルとの比較があります。ここでは **XL-Sum ja** という評価指標に注目してみます。

2つの評価テーブルを見ると **XL-Sum ja** については **Stability AI** のモデルが高いスコアを示していますので、今回は **Japanese Stable LM Base Gamma 7B** を基盤モデルとしましょう。

### 2. データセットの選定
次に先ほどの基盤モデルを今回のタスクにファインチューニングするためのデータセットが必要です。

適応させるタスクによってはそのほとんどを一から作成する必要がありますが、今回は既存の公開データセットを使用します。

[databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k)はDatabricksによって公開された、同社の社員によって人手で作成された複数タスクの指示応答データセットです。この中のsummarizationカテゴリを抽出してファインチューニングに使用しましょう。

また、有志によって上記のデータセットは日本語訳されていますので、今回は[こちら](https://huggingface.co/datasets/kunishou/databricks-dolly-15k-ja)を使用します。

### 3. データセットの前処理
さて、今回はSFT、つまり教師ありデータで基盤モデルをファインチューニングしていきます。

要するにデータセットの各レコードを *モデルへの入力* と *期待されるモデルの出力* を含んだプロンプトフォーマットに当てはめていく必要があります。
ここでは日本語指示調整モデルで比較的採用率の高い下記のプロンプトフォーマットを使用しましょう。
```
### 指示
{指示内容}

### 入力
{要約する文章}

### 応答
{期待される要約}
```

### 4. 基盤モデルの出力確認
ここで、先ほど選択したモデルの今回のタスクに対する検証を行ってみましょう。

### 5. ファインチューニングの設定
では実際に選択した基盤モデルとデータセットを用いて微調整を行っていこうと思いますが
今回選択したのは7Bのパラメーターサイズです。その全てのパラメーターを微調整するとなるとハード要件がぐっとあがります。
そこで今回はLoRAという一部のパラメーターの微調整で全パラメーターの微調整と同等の精度を期待できる手法を用います。
それによって学習に必要なメモリを削減、また、学習時間の削減にもなります。

ファインチューニングの複雑さを排除したライブラリや、最適化を行うフレームワークが多く存在しますが
今回はHugging Faceが提供するTRLとPEFTを使用して微調整を行っていきます。
TRLはSFTを行う学習フローを提供し、PEFTは先述のLoRAをSFTに適用するために必要となります。

TRLのSFTTrainer、PEFTともに多くのパラメーターを適用できますので
それぞれコンフィグオブジェクトを定義していきましょう。
全てを説明するには余白が足りませんので主要なパラメーターのみ説明します。
こうしたいんだけど...というものがあれば実際に公式ドキュメントやソースコードを読むとぴったりのパラメータがあるかもしれません。

### 6. ファインチューニング
ではコンフィグオブジェクトが出来ましたのでそれをSFTTrainerに渡して学習を開始します。

### 7. ファインチューニングモデルの出力確認
では推論してみましょう。

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig

dataset = load_dataset("timdettmers/openassistant-guanaco", split="train")

model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

instruction_template = "### Human:"
response_template = "### Assistant:"
collator = DataCollatorForCompletionOnlyLM(instruction_template=instruction_template, response_template=response_template, tokenizer=tokenizer, mlm=False)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

trainer = SFTTrainer(
    model,
    args=SFTConfig(output_dir="/tmp"),
    train_dataset=dataset,
    data_collator=collator,
    peft_config=peft_config
)

trainer.train()
```

:::note
- **This is a DRAFT!!**
- このハンズオンをなぜするのかを明確にしておきたい
    - 何のために？が分からないとただ手を動かすだけになってしまうかも
- 上から下への流れを重要視する、細かく説明したほうがいいのだろうが、ユーザーの手を止めすぎないように
- ベースモデルとは？
- プロンプトって何を指している？
- LoRAとは？(パラメータ？)
    - どうやって一部分だけを更新する？
- 学習時間の目安は？
    - 何によって変わる？
- 何のパラメータをいじれば改善できるの？
- SFTが何で、LoRAは何？というところから入る？
- なるべく公式リポジトリ推奨の方法で
- WandBとかの管理ツールは含める？
    - 無いほうが目的に対して理解がしやすいと思うが、いざ管理するときにまた障壁にならないか？
- notebook形式の方がパッと試しやすい？
- Hugging Faceへのモデルアップロードの方法もあったらいいかも
- 参考
    - [PEFT公式ドキュメント](https://huggingface.co/docs/peft/index)
    - [TRL公式ドキュメント](https://huggingface.co/docs/trl/index)
    - [TRL SFTのスクリプト例](https://github.com/huggingface/trl/blob/main/examples/scripts/sft.py)
:::
