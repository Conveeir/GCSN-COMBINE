data_dir = "Dataset/LogiQA"
model_type = "Roberta"
assert model_type == 'Roberta'
SEP_TOKEN = '</s>'
CLS_TOKEN = '<s>'
NODE_SEP_TOKEN = '~'

tokenizer = None
model_args = None
data_args = None
train_args = None

node_intervals_padding_id = 10086
max_seq_length = 256
max_edge_num = 40
max_node_num = 20

max_tok_len = 200
