from itertools import islice
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# 基础特殊token
base_special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
content_special_tokens=["<begin_of_article>","<end_of_article>",
"<begin_of_paragraph>","<end_of_paragrph>","<begin_of_centence>","<end_of_centence>",
"<begin_of_conversation>","<end_of_conversation>","<begin_of_user>","<end_of_user>",
"<begin_of_assistant>","<end_of_assistant>","<begin_of_bot>","<end_of_bot>",
"<begin_of_thinking>","<end_of_thinking>"]
# 10个预留token（可根据需要自定义）
reserved_tokens = [
    "<reserved_1>", "<reserved_2>", "<reserved_3>", 
    "<reserved_4>", "<reserved_5>", "<reserved_6>", 
    "<reserved_7>", "<reserved_8>", "<reserved_9>", 
    "<reserved_10>"
]

# 合并所有特殊token
all_special_tokens = base_special_tokens +content_special_tokens+ reserved_tokens

# 初始化分词器（设置unk_token）
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

# 配置训练器
trainer = BpeTrainer(
    vocab_size=8000,
    special_tokens=all_special_tokens
)

dataset_path = "/kaggle/working/wiki.txt"

def line_iterator(path, skip=0):
    with open(path, "r", encoding="utf-8") as f:
        for line in islice(f, skip, None):
            yield line.strip()

# 训练分词器
tokenizer.train_from_iterator(line_iterator(dataset_path, 0), trainer=trainer)

# 保存分词器
output_path = "/kaggle/working/tokenizer.json"
tokenizer.save(output_path)

# 验证特殊token是否添加成功
print("特殊token列表：")
for i, token in enumerate(all_special_tokens):
    token_id = tokenizer.token_to_id(token)
    print(f"{i+1:2d}. {token:20} -> ID: {token_id}")

print(f"\n总特殊token数量: {len(all_special_tokens)}")
print(f"词汇表大小: {tokenizer.get_vocab_size()}")