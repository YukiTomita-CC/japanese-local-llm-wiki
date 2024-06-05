---
displayed_sidebar: handson
description: LoRAを使ったラベルありファインチューニング
---

# SFT with LoRA
:::note
- **This is a DRAFT!!**
- このハンズオンをなぜするのかを明確にしておきたい
    - 何のために？が分からないとただ手を動かすだけになってしまうかも
- 上から下への流れを重要視する、細かく説明したほうがいいのだろうが、ユーザーの手を止めすぎないように
- SFTが何で、LoRAは何？というところから入る？
- なるべく公式リポジトリ推奨の方法で
- 参考
    - [PEFT公式ドキュメント](https://huggingface.co/docs/peft/index)
    - [TRL公式ドキュメント](https://huggingface.co/docs/trl/index)
    - [TRL SFTのスクリプト例](https://github.com/huggingface/trl/blob/main/examples/scripts/sft.py)
:::

## 目的
- ベースモデルに対してラベルありデータでのファインチューニングを行う
- 与えるプロンプトとそれに対して求められる出力のペアで学習する
- LoRAは全パラメータに対して一部分のみを更新する
- それによって学習に必要なメモリを削減、また、学習時間の削減にもなる

:::note
- ベースモデルとは？
- プロンプトって何を指している？
- LoRAとは？(パラメータ？)
    - どうやって一部分だけを更新する？
- 学習時間の目安は？
    - 何によって変わる？
- 何のパラメータをいじれば改善できるの？
:::

## タスクの設定
- llama2にするか、mistralにするか。。
    - llama2の方がベースモデルは多い、ただライセンス問題が絡んでくる
    - その点mistralはapache-2.0だが、Swallow-MSかRakutenかどっちかしかない
- チューニング後の変化が分かりやすいほうがいいけどなるべく実用的なタスク
- チューニング前、チューニング後の出力の比較はmust

## コード
:::note
- TRLとPEFTは何が違うの？それぞれ何をするためのライブラリ？
- WandBとかの管理ツールは含める？
    - 無いほうが目的に対して理解がしやすいと思うが、いざ管理するときにまた障壁にならないか？
- notebook形式の方がパッと試しやすい？
- Hugging Faceへのモデルアップロードの方法もあったらいいかも
:::

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
