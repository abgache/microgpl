import torch
import torch.nn as nn
import json
import os
from math import ceil, sin, cos, sqrt
from collections import Counter
from scripts.time_log import time_log_module as tlm
from scripts.byte_pair_encoder import data2tokens
from scripts.graph import plot_attention_matrix

class tokenizer(): # Fully functional
    def __init__(self, logger, tokenizer_config):
        self.logger = logger
        tokenizer_config = tokenizer_config
        self.vocab_size = tokenizer_config.get("vocab_size", 10000)
        self.path = tokenizer_config.get("path", "model/tokenizer.json")
        self.special_tokens = tokenizer_config.get("special_tokens", {})
        self.tokens = {} # e.g. {"token": index}
        self.logger.log(f"Tokenizer initialized with vocab size {self.vocab_size}.", v=True, Wh=True, mention=False)
    
    def tokenize(self, text):
        # Tokens format: self.tokens = {} # e.g. {"token": index}
        tokens = []
        i = 0
        # Trie les tokens du vocab du plus long au plus court pour BPE greedy
        vocab_tokens = sorted(self.tokens.keys(), key=len, reverse=True)

        while i < len(text):
            match = None
            for token in vocab_tokens:
                if text.startswith(token, i):
                    match = token
                    break
            if match:
                tokens.append(self.tokens[match])
                i += len(match)
            else:
                tokens.append(self.tokens.get("<unk>", 0))
                i += 1
        return tokens
    
    def detokenize(self, token_ids):
        id_to_token = {index: token for token, index in self.tokens.items()}
        tokens = [id_to_token.get(token_id, "<unk>") for token_id in token_ids]
        text = " ".join(tokens)
        return text
    
    def create_vocab(self, dataset):
        unique_tokens = data2tokens(dataset, vocab_size=((int(self.vocab_size))-4)) # Way less optimized than the old data.split but WAY more effective for MGPL
        self.tokens = {token: idx+len(self.special_tokens)+1 for idx, token in enumerate(unique_tokens)}
        for special_token, index in self.special_tokens.items():
            self.tokens[special_token] = index
        self.logger.log(f"Vocabulary created with {len(self.tokens)} tokens.", v=True, Wh=True, mention=False)

    def save_vocab(self):
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.tokens, f, ensure_ascii=False, indent=4)
            self.logger.log(f"Vocabulary saved to {self.path}.", v=True, Wh=True, mention=False)
        except Exception as e:
            self.logger.log(f"Error saving vocabulary to {self.path}: {e}", v=False, Wh=True, mention=True)
            raise ValueError(f"Error saving vocabulary to {self.path}: {e}")
    
    def load_vocab(self):
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                self.tokens = json.load(f)
            self.logger.log(f"Vocabulary loaded from {self.path}.", v=True, Wh=True, mention=False)
        except Exception as e:
            self.logger.log(f"Error loading vocabulary from {self.path}: {e}", v=False, Wh=True, mention=True)
            raise ValueError(f"Error loading vocabulary from {self.path}: {e}")
    
    def vocab_status(self): # If a tokenizer (tokenizer.json) exists and is not empty
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = f.read()
                return not (data == None or data == "{}")
        except:
            return False

class embedding():
    def __init__(self, logger, device, tokenizer, embedding_config):
        self.logger = logger
        self.full_model = None
        self.device = device
        self.embedding_table = []
        self.tokenizer = tokenizer
        embedding_config = embedding_config
        self.vector_dim = embedding_config.get("vector_dim", 128)
        self.main_model_path = embedding_config.get("main_model_path", "model/embedding_model.pth")
        self.json_table_path = embedding_config.get("json_table_path", "model/embedding_table.json")
        self.dnn_config = embedding_config.get("dnn", {})
        if not self.tokenizer.vocab_size == self.dnn_config.get("input_size", {}):
            self.logger.log("Tokenizer vocabulary size is not equal to the input size of the embedding dnn. Cannot initialize embedding module.", v=False, Wh=True, mention=True)
            raise ValueError(f"{tlm()} Tokenizer vocabulary size is not equal to the input size of the embedding dnn. Cannot initialize embedding module.")
        self.logger.log("Embedding module initialized.", v=True, Wh=True, mention=False)
        # il faut pas save le model de base, on save juste une db des vecteurs {"token": index}
    
    def check_saved_embedding_table(self):
        try:
            with open(self.json_table_path, "r", encoding="utf-8") as f:
                data = f.read()
                return len(data) > 0
        except:
            return False

    def load_embedding_table(self):
        try:
            with open(self.json_table_path, "r", encoding="utf-8") as f:
                self.embedding_table = json.load(f)
            self.logger.log(f"Embedding table loaded from {self.json_table_path}.", v=True, Wh=True, mention=False)
            return self.embedding_table
        except Exception as e:
            self.logger.log(f"Error loading embedding table from {self.json_table_path}: {e}", v=False, Wh=True, mention=True)
            raise ValueError(f"{tlm()} Error loading embedding table from {self.json_table_path}: {e}")
    
    def save_embedding_table(self):
        try:
            with open(self.json_table_path, "w", encoding="utf-8") as f:
                json.dump(self.embedding_table, f, ensure_ascii=False, indent=4)
            self.logger.log(f"Embedding table saved to {self.json_table_path}.", v=True, Wh=True, mention=False)
        except Exception as e:
            self.logger.log(f"Error saving embedding table to {self.json_table_path}: {e}", v=False, Wh=True, mention=True)
            raise ValueError(f"{tlm()} Error saving embedding table to {self.json_table_path}: {e}")

    def create_embedding_model(self):
        # Placeholder for creating the embedding model based on dnn_config
        # DNN : Input : vocab_size (10,000), 2nd Layer (Linear Func) : vector_dim (128), 3rd layer (Sigmoid) : Transition layer (512), Output Layer : vocab_size (10,000) with Softmax
        self.full_model = nn.Sequential(
            nn.Linear(self.tokenizer.vocab_size, int(self.vector_dim)),             # Hidden Layer 1
            nn.Linear(int(self.vector_dim), int(self.vector_dim)*4),               # Hidden Layer 2
            nn.Sigmoid(),
            nn.Linear(int(self.vector_dim)*4, self.tokenizer.vocab_size),                  # Output Layer
            nn.Softmax(dim=1)
        )
        self.logger.log(f"Embedding model created based on DNN configuration. The embedding model has {str(sum(p.numel() for p in self.full_model.parameters()))} parameters.", v=True, Wh=True, mention=False)

    def train_embedding_model(self, dataset, json_data_path):
        if not self.tokenizer.vocab_status():
            self.logger.log("Tokenizer vocabulary not found. Cannot create embedding table.", v=False, Wh=True, mention=True)
            raise ValueError(f"{tlm()} Tokenizer vocabulary not found. Cannot create embedding table.")
        # Placeholder for training the embedding model
        # format dataset into input-output pairs for training
        if os.path.exists(json_data_path) and os.path.getsize(json_data_path) > 0:
            self.logger.log(f"Training data file {json_data_path} already exists. Skipping data preparation step and loading data.", v=True, Wh=True, mention=False)
            try:
                with open(json_data_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    input_data = [data["input_data"][str(i)] for i in range(len(data["input_data"]))]
                    output_data = [data["output_data"][str(i)] for i in range(len(data["output_data"]))]
                self.logger.log(f"Training data loaded from {json_data_path}.", v=True, Wh=True, mention=False)
            except Exception as e:
                self.logger.log(f"Error loading training data from {json_data_path}: {e}", v=False, Wh=True, mention=True)
                raise ValueError(f"{tlm()} Error loading training data from {json_data_path}: {e}")
        else:
            if isinstance(dataset, str):
                tokenized_data = self.tokenizer.tokenize(dataset)
            else:
                tokenized_data = [self.tokenizer.tokenize(data) for data in dataset]
            del dataset

            # Create input/output data
            # E.g. : "I love programming in Python using PyTorch" → [I, love, programming, in, Python, using, PyTorch]
            # For the input "Python", 4examples : "programming", "in", "using", "PyTorch" => The two tokens before and after the target token (For 1st token, just add a <UNK> token and for the last token a <EOS> token)
            input_data = []  # One-hot encoded vectors of target tokens
            output_data = [] # One-hot encoded vectors of context tokens

            self.logger.log("Preparing training data for embedding model...", v=True, Wh=True, mention=False)
            for idx, target_token_id in enumerate(tokenized_data):
                # Context tokens
                context_token_ids = [
                    tokenized_data[idx - 2] if idx - 2 >= 0 else 2,
                    tokenized_data[idx - 1] if idx - 1 >= 0 else 2,
                    tokenized_data[idx + 1] if idx + 1 < len(tokenized_data) else 4,
                    tokenized_data[idx + 2] if idx + 2 < len(tokenized_data) else 4
                ]
                input_data.extend([target_token_id] * 4)
                output_data.extend(context_token_ids)

            if not len(input_data) == len(output_data):
                self.logger.log(f"Input and output data lengths do not match. Cannot train embedding model. Input size : {len(input_data)}, Output size : {len(output_data)}", v=False, Wh=True, mention=True)
                raise ValueError(f"{tlm()} Input and output data lengths do not match. Cannot train embedding model. Input size : {len(input_data)}, Output size : {len(output_data)}")
            del tokenized_data
            try:
                with open(json_data_path, "w", encoding="utf-8") as f:
                    in_data = {}
                    out_data = {}
                    for i in range(len(input_data)):
                        in_data[i] = input_data[i]
                        out_data[i] = output_data[i]
                    data2save = {"input_data": in_data, "output_data": out_data}
                    json.dump(data2save, f, ensure_ascii=False, indent=4)
                self.logger.log(f"Trainning data saved to {json_data_path}.", v=True, Wh=True, mention=False)
            except:
                self.logger.log(f"Error saving trainning data to {json_data_path}: {Exception}", v=False, Wh=True, mention=True)
                raise ValueError(f"Error saving trainning data to {json_data_path}: {Exception}")
        self.logger.log("Transforming the data to one-hot vectors...", v=True, Wh=True, mention=False)
        
        self.dnn_config = self.dnn_config
        self.num_epochs = self.dnn_config.get("num_epochs", 10)
        self.num_batches = self.dnn_config.get("batch_size", 32)
        self.learning_rate = self.dnn_config.get("learning_rate", 0.001)
        self.criterion = torch.nn.KLDivLoss(reduction='batchmean')
        self.optimizer = torch.optim.Adam(self.full_model.parameters(), lr=self.learning_rate)

        if os.path.exists(self.main_model_path) and os.path.getsize(self.main_model_path) > 0:
            self.logger.log(f"Pretrained embedding model found at {self.main_model_path}. Loading model...", v=True, Wh=True, mention=False)
            self.full_model.load_state_dict(torch.load(self.main_model_path, map_location=self.device))
            self.logger.log("Pretrained embedding model loaded successfully.", v=True, Wh=True, mention=False)
        else:
            self.logger.log("No pretrained embedding model found. Starting training from scratch.", v=True, Wh=True, mention=False)  
            # Generating 50,000 samples at a time to avoid memory issues then trainning on then deleting them and repeating until all data is processed
            data_size = len(input_data) # A single One Hot vector is of size vocab_size (10,000) soo =~ 20ko bcs there is the input and the output
            chunk_size = 25000 # 25,000 samples = 500mo of data in RAM at once (25,000 * 20ko)
            num_chunks = (data_size + chunk_size - 1) // chunk_size
            TMP_cycles = 0
            self.logger.log(f"Starting trainning, number of chunks: {num_chunks}, data size: {data_size}", v=True, Wh=True, mention=False)
            while TMP_cycles < num_chunks:
                # Create One-Hot Encoded vectors for each token
                input_oh_data = []
                output_oh_data = []
                TMP_cycles += 1
    
                start_idx = (TMP_cycles - 1) * chunk_size
                end_idx = min(TMP_cycles * chunk_size, data_size)
                for i in range(start_idx, end_idx):
                    # Input One-Hot
                    if input_data[i] == None:
                        input_data[i] = 2
                    token = input_data[i]
                    one_hot = [0] * self.tokenizer.vocab_size
                    one_hot[token-1] = 1
                    input_oh_data.append(one_hot)
    
                    # Output One-Hot 2 = UNK 
                    if output_data[i] == None:
                        output_data[i] = 2
                    token = output_data[i]
                    one_hot = [0] * self.tokenizer.vocab_size
                    one_hot[token-1] = 1
                    output_oh_data.append(one_hot)
    
                # torch.tensor
                input_tensor = torch.tensor(input_oh_data, dtype=torch.float32).to(self.device)
                output_tensor = torch.tensor(output_oh_data, dtype=torch.float32).to(self.device)
                del one_hot
    
                # Train the model
                self.full_model.to(self.device)
    
                self.full_model.train()
    
                for epoch in range(self.num_epochs):
                    total_loss = 0.0
    
                    for i in range(self.num_batches):
                        start = i * self.num_batches
                        end = start + self.num_batches
    
                        batch_x = input_tensor[start:end]
                        batch_y = output_tensor[start:end]
    
                        # Reset gradients
                        self.optimizer.zero_grad()
    
                        # Forward
                        outputs = torch.log(self.full_model(batch_x) + 1e-9)
    
                        # Loss
                        loss = self.criterion(outputs, batch_y)
    
                        # Backprop
                        loss.backward()
                        self.optimizer.step()
    
                        total_loss += loss.item()
    
                    avg_loss = total_loss / max(1, self.num_batches)
                    self.logger.log(
                        f"Epoch [{epoch+1}/{self.num_epochs}] - Loss: {avg_loss:.4f}",
                        v=True, Wh=True, mention=False
                    )
                del input_tensor, output_tensor
                torch.cuda.empty_cache()
    
    
            self.logger.log("Training completed successfully.", v=True, Wh=True, mention=False)
    
            torch.save(self.full_model.state_dict(), self.main_model_path)

        # Create embedding table
        for index in range(self.tokenizer.vocab_size):
            token = int(index)+1
            token_oh = []
            one_hot = [0] * self.tokenizer.vocab_size
            one_hot[token-1] = 1
            token_oh.append(one_hot)
            x = torch.tensor(token_oh, dtype=torch.float32)
            x = x.to(self.device)
            self.full_model.to(self.device)
            upto_layer = 1
            for i, layer in enumerate(self.full_model):
                x = layer(x)
                if i == upto_layer:
                    break
            self.embedding_table += [(token, x.detach().tolist())]

class SPE():
    def __init__(self, device):
        self.device = device
    
    def vector2spe_vector(self, input_vector, position):
        input_vector = input_vector.tolist() if torch.is_tensor(input_vector) else input_vector
        spe_vector = []
        for i in range(len(input_vector)):
            if i % 2 == 0:
                spe_vector.append(torch.sin(position / (10000 ** ((2 * i) / len(input_vector)))))
            else:
                spe_vector.append(torch.cos(position / (10000 ** ((2 * (i + 1)) / len(input_vector)))))
        spe_vector = torch.tensor(spe_vector, dtype=torch.float32).to(self.device)
        return spe_vector
    
    def vector_list2spe_vector_list(self, input_vector_list):
        spe_vector_list = []
        for pos in range(len(input_vector_list)):
            spe_vector = self.vector2spe_vector(input_vector_list[pos], pos)
            spe_vector_list.append(spe_vector)
        del input_vector_list
        return spe_vector_list

class attention_head():
    def __init__(self, logger, embedding, SPE, attention_config):
        self.logger = logger
        self.embedding = embedding
        self.tokenizer = self.embedding.tokenizer
        self.attention_config = attention_config
        self.context_window = self.attention_config.get("context_window", 64)
        self.attention_matrix = None # Single attention head
        self.SPE = SPE
        self.device = self.embedding.device
        self.masking_value = self.attention_config.get("masking_value", -1e9) # Change for -1e4 if ur using float16

        # Weight matrices
        # DO NOT to transform them into torch tensors
        self.wq = None
        self.wk = None
        self.wv = None

        # Weight matrices paths
        self.wq_path = self.attention_config.get("wq_path", "model/wq.json")
        self.wk_path = self.attention_config.get("wk_path", "model/wk.json")
        self.wv_path = self.attention_config.get("wv_path", "model/wv.json")

    def get_wq(self):
        if os.path.exists(self.wq_path) and os.path.getsize(self.wq_path) > 0:
            try:
                with open(self.wq_path, "r", encoding="utf-8") as f:
                    self.wq = torch.tensor(json.load(f), dtype=torch.float32).to(self.device)
                self.logger.log(f"Wq matrix loaded from {self.wq_path}.", v=True, Wh=True, mention=False)
            except Exception as e:
                self.logger.log(f"Error loading Wq matrix from {self.wq_path}: {e}", v=False, Wh=True, mention=True)
                raise ValueError(f"{tlm()} Error loading Wq matrix from {self.wq_path}: {e}")
        else:
            self.logger.log(f"Wq matrix file {self.wq_path} not found. Trainning new Wq matrix...", v=False, Wh=True, mention=True)
            self.wq = torch.randn((self.embedding.vector_dim, self.embedding.vector_dim), dtype=torch.float32).to(self.device)
            # save it
            try:
                with open(self.wq_path, "w", encoding="utf-8") as f:
                    json.dump(self.wq.tolist(), f, ensure_ascii=False, indent=4)
                self.logger.log(f"Wq matrix saved to {self.wq_path}.", v=True, Wh=True, mention=False)
            except Exception as e:
                self.logger.log(f"Error saving Wq matrix to {self.wq_path}: {e}", v=False, Wh=True, mention=True)
                raise ValueError(f"{tlm()} Error saving Wq matrix to {self.wq_path}: {e}")
    
    def get_wk(self):
        if os.path.exists(self.wk_path) and os.path.getsize(self.wk_path) > 0:
            try:
                with open(self.wk_path, "r", encoding="utf-8") as f:
                    self.wk = torch.tensor(json.load(f), dtype=torch.float32).to(self.device)
                self.logger.log(f"Wk matrix loaded from {self.wk_path}.", v=True, Wh=True, mention=False)
            except Exception as e:
                self.logger.log(f"Error loading Wk matrix from {self.wk_path}: {e}", v=False, Wh=True, mention=True)
                raise ValueError(f"{tlm()} Error loading Wk matrix from {self.wk_path}: {e}")
        else:
            self.logger.log(f"Wk matrix file {self.wk_path} not found. Trainning new Wk matrix...", v=False, Wh=True, mention=True)
            self.wk = torch.randn((self.embedding.vector_dim, self.embedding.vector_dim), dtype=torch.float32).to(self.device)
            # save it
            try:
                with open(self.wk_path, "w", encoding="utf-8") as f:
                    json.dump(self.wk.tolist(), f, ensure_ascii=False, indent=4)
                self.logger.log(f"Wk matrix saved to {self.wk_path}.", v=True, Wh=True, mention=False)
            except Exception as e:
                self.logger.log(f"Error saving Wk matrix to {self.wk_path}: {e}", v=False, Wh=True, mention=True)
                raise ValueError(f"{tlm()} Error saving Wk matrix to {self.wk_path}: {e}")

    def get_wv(self):
        if os.path.exists(self.wv_path) and os.path.getsize(self.wv_path) > 0:
            try:
                with open(self.wv_path, "r", encoding="utf-8") as f:
                    self.wv = torch.tensor(json.load(f), dtype=torch.float32).to(self.device)
                self.logger.log(f"Wv matrix loaded from {self.wv_path}.", v=True, Wh=True, mention=False)
            except Exception as e:
                self.logger.log(f"Error loading Wv matrix from {self.wv_path}: {e}", v=False, Wh=True, mention=True)
                raise ValueError(f"{tlm()} Error loading Wv matrix from {self.wv_path}: {e}")
        else:
            self.logger.log(f"Wv matrix file {self.wv_path} not found. Trainning new Wv matrix...", v=False, Wh=True, mention=True)
            self.wv = torch.randn((self.embedding.vector_dim, self.embedding.vector_dim), dtype=torch.float32).to(self.device)

            # save it
            try:
                with open(self.wv_path, "w", encoding="utf-8") as f:
                    json.dump(self.wv.tolist(), f, ensure_ascii=False, indent=4)
                self.logger.log(f"Wv matrix saved to {self.wv_path}.", v=True, Wh=True, mention=False)
            except Exception as e:
                self.logger.log(f"Error saving Wv matrix to {self.wv_path}: {e}", v=False, Wh=True, mention=True)
                raise ValueError(f"{tlm()} Error saving Wv matrix to {self.wv_path}: {e}")

    def embed2query(self, input_embedding):
        if isinstance(input_embedding, list):
            input_embedding = torch.tensor(input_embedding, dtype=torch.float32).to(self.device)
        return self.wq @ input_embedding
    
    def embed2key(self, input_embedding):
        if isinstance(input_embedding, list):
            input_embedding = torch.tensor(input_embedding, dtype=torch.float32).to(self.device)
        return self.wk @ input_embedding

    def embed2value(self, input_embedding):
        if isinstance(input_embedding, list):
            input_embedding = torch.tensor(input_embedding, dtype=torch.float32).to(self.device)
        return self.wv @ input_embedding

    def create_attention_matrix(self, input_embeddings, spe_encoded=False):
        if not spe_encoded:
            input_embeddings = self.SPE.vector_list2spe_vector_list(input_embeddings)

        # Calculate queries & keys
        queries = torch.tensor([self.embed2query(emb) for emb in input_embeddings], dtype=torch.float32).to(self.device)
        keys = torch.tensor([self.embed2key(emb) for emb in input_embeddings], dtype=torch.float32).to(self.device)

        scores = (queries @ keys.T) / math.sqrt(queries.size(-1))
        scores = scores.masked_fill(torch.triu(torch.ones_like(scores), 1).bool(), self.masking_value)
        self.attention_matrix = torch.softmax(scores, dim=-1)
        plot_attention_matrix(self.attention_matrix, "data/attention.png")

        # Apply softmax to get attention weights
        #self.attention_matrix = torch.nn.functional.softmax(torch.tensor(matrix, dtype=torch.float32).to(self.device), dim=-1)
    
    def get_new_vector(self, position):
        if self.attention_matrix is None:
            self.logger.log("Attention matrix not created yet. Cannot get new vector.", v=False, Wh=True, mention=True)
            raise ValueError(f"{tlm()} Attention matrix not created yet. Cannot get new vector.")
        attention_weights = self.attention_matrix[position]
        value_vectors = torch.tensor([self.embed2value(emb) for emb in self.embedding.embedding_table], dtype=torch.float32).to(self.device)
        new_vector = attention_weights @ value_vectors
        return new_vector

        
class FNN():
    def __init__(self, logger, embedding, ffn_config):
        self.logger = logger
        self.embedding = embedding
        self.ffn_config = ffn_config
        self.model_path = self.ffn_config.get("model_path", "model/ffn.pth")
        self.input_size = self.ffn_config.get("input_size", 256)
        self.num_epochs = self.ffn_config.get("num_epochs", 25)
        self.batch_size = self.ffn_config.get("batch_size", 32)
        self.learning_rate = self.ffn_config.get("learning_rate", 0.001)
        self.model = nn.Sequential(
            nn.Linear(self.embedding.vector_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Linear(4096, self.embedding.tokenizer.vocab_size),
            nn.Softmax(dim=1)
        )
        self.logger.log("Feedforward Neural Network initialized.", v=True, Wh=True, mention=False)
    
    def train_ffn(self, x, y):
        pass
