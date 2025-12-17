import torch
import torch.nn as nn
import json
from scripts.time_log import time_log_module as tlm

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
        tokens = text.split() # Simple whitespace tokenizer
        token_ids = [self.tokens.get(token, self.tokens.get("<unk>")) for token in tokens]
        return token_ids
    
    def detokenize(self, token_ids):
        id_to_token = {index: token for token, index in self.tokens.items()}
        tokens = [id_to_token.get(token_id, "<unk>") for token_id in token_ids]
        text = " ".join(tokens)
        return text
    
    def create_vocab(self, dataset):
        unique_tokens = set(dataset.split())
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
        self.logger.log(f"Embedding model created based on DNN configuration. The embedding model has {str(self.full_model.parameters())} parameters.", v=True, Wh=True, mention=False)

    def train_embedding_model(self, dataset):
        if not self.tokenizer.vocab_status():
            self.logger.log("Tokenizer vocabulary not found. Cannot create embedding table.", v=False, Wh=True, mention=True)
            raise ValueError(f"{tlm()} Tokenizer vocabulary not found. Cannot create embedding table.")
        # Placeholder for training the embedding model
        # format dataset into input-output pairs for training
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

        
        for idx, target_token_id in enumerate(tokenized_data):
            # Context tokens
            context_token_ids = [
                tokenized_data[idx - 2] if idx - 2 >= 0 else self.tokenizer.tokens.get("<unk>"),
                tokenized_data[idx - 1] if idx - 1 >= 0 else self.tokenizer.tokens.get("<unk>"),
                tokenized_data[idx + 1] if idx + 1 < len(tokenized_data) else self.tokenizer.tokens.get("<eos>"),
                tokenized_data[idx + 2] if idx + 2 < len(tokenized_data) else self.tokenizer.tokens.get("<eos>")
            ]
            input_data.extend([target_token_id] * 4)
            output_data.extend(context_token_ids)

        if not len(input_data) == len(output_data):
            self.logger.log(f"Input and output data lengths do not match. Cannot train embedding model. Input size : {len(input_data)}, Output size : {len(output_data)}", v=False, Wh=True, mention=True)
            raise ValueError(f"{tlm()} Input and output data lengths do not match. Cannot train embedding model. Input size : {len(input_data)}, Output size : {len(output_data)}")
        del tokenized_data

        # Create One-Hot Encoded vectors for each token
        input_oh_data = []
        for token_ids in input_data:
            print(f"{str(self.tokenizer.vocab_size)}  vs {str(token_ids)}")
            one_hot = [0] * self.tokenizer.vocab_size
            one_hot[token_ids] = 1
            input_oh_data.append(one_hot)
        
        output_oh_data = []
        for token_ids in output_data:
            one_hot = [0] * self.tokenizer.vocab_size
            one_hot[token_ids] = 1
            output_oh_data.append(one_hot)

        # torch.tensor
        input_tensor = torch.tensor(input_oh_data, dtype=torch.float32).to(self.device)
        output_tensor = torch.tensor(output_oh_data, dtype=torch.float32).to(self.device)
        del input_oh_data
        del output_oh_data
        del one_hot

        # Train the model
        self.dnn_config = self.dnn_config
        self.num_epochs = self.dnn_config.get("num_epochs", 10)
        self.num_batches = self.dnn_config.get("batch_size", 32)
        self.learning_rate = self.dnn_config.get("learning_rate", 0.001)
        self.criterion = torch.nn.MSELoss() # Mean Squared Error Loss for regression
        self.optimizer = torch.optim.Adam(self.full_model.parameters(), lr=self.learning_rate) 

        self.full_model.to(self.device)

        self.full_model.train()

        for epoch in range(self.num_epochs):
            total_loss = 0.0

            for i in range(num_batches):
                start = i * batch_size
                end = start + batch_size

                batch_x = input_oh_data[start:end]
                batch_y = output_oh_data[start:end]

                # Reset gradients
                self.optimizer.zero_grad()

                # Forward
                outputs = self.full_model(batch_x)

                # Loss
                loss = self.criterion(outputs, batch_y)

                # Backprop
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / max(1, num_batches)
            self.logger.log(
                f"Epoch [{epoch+1}/{self.num_epochs}] - Loss: {avg_loss:.4f}",
                v=True, Wh=True, mention=False
            )

        self.logger.log("Training completed successfully.", v=True, Wh=True, mention=False)

        # Create embedding table
        for index in self.tokenizer.vocab_size:
            token = int(index)
            token_oh = []
            one_hot = [0] * self.tokenizer.vocab_size
            one_hot[token] = 1
            token_oh.append(one_hot)
            embed = self.full_model(torch.tensor(token_oh, dtype=torch.float32).to(self.device), upto_layer=1)
            self.embedding_table += [(token, embed)]


class model():
    def __init__(self, logger, embedding, context_window=64):
        self.logger = logger
        self.embedding = embedding
        self.tokenizer = self.embedding.tokenizer
        self.context_window = context_window
        self.attention_matrix = None # Single attention head
    
    def positional_encoding(self, position, d_model=None): # Avoir le vecteur de position qu'on vas après additionner au vecteur du mot
        if d_model is None:
            d_model = self.embedding.vector_dim
        pe = torch.zeros(position, d_model)
        for pos in range(position):
            for i in range(0, d_model, 2):
                pe[pos, i] = torch.sin(pos / (10000 ** ((2 * i)/d_model)))
                if i + 1 < d_model:
                    pe[pos, i + 1] = torch.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        return pe
    
    def encode_vector_position(self, input_vectors: list): # Encode la position des vecteurs d'une liste de vecteurs
        pe_vectors = []
        for pos in range(len(input_vectors)):
            pe = self.positional_encoding(pos)
            pe_vectors.append(input_vectors[pos] + pe)
        return pe_vectors # vrm pas un code de tigre

    def create_attention_matrix(self):
        pass


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
