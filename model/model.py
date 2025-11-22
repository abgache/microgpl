import torch
import json

class tokenizer(): # Simply Byte-Pair Encoding tokenizer
    def __init__(self, logger, tokenizer_config):
        self.logger = logger
        tokenizer_config = json.load(tokenizer_config)
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
        self.tokens = {token: idx for idx, token in enumerate(unique_tokens)}
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
                return len(data) > 0
        except:
            return False

class embedding():
    def __init__(self, logger, tokenizer, embedding_config):
        self.logger = logger
        self.full_model = None
        self.embedding_table = {}
        self.tokenizer = tokenizer
        embedding_config = json.load(embedding_config)
        self.vector_dim = embedding_config.get("vector_dim", 128)
        self.main_model_path = embedding_config.get("main_model_path", "model/embedding_model.pth")
        self.json_table_path = embedding_config.get("json_table_path", "model/embedding_table.json")
        self.dnn_config = embedding_config.get("dnn", {})
        if not self.tokenizer.vocab_size == self.dnn_config.get("input_size", {}):
            self.logger.log("Tokenizer vocabulary size is not equal to the input size of the embedding dnn. Cannot initialize embedding module.", v=False, Wh=True, mention=True)
            raise ValueError(f"{tlm()} Tokenizer vocabulary size is not equal to the input size of the embedding dnn. Cannot initialize embedding module.")
        self.logger.log("Embedding module initialized.", v=True, Wh=True, mention=False)
        # il faut pas save le model de base, on save juste une db des vecteurs {token_id: {vector}}
    
    def load_embedding_table(self):
        try:
            with open(self.json_table_path, "r", encoding="latin-1") as f:
                self.embedding_table = json.load(f)
            self.logger.log(f"Embedding table loaded from {self.json_table_path}.", v=True, Wh=True, mention=False)
            return embedding_table
        except Exception as e:
            self.logger.log(f"Error loading embedding table from {self.json_table_path}: {e}", v=False, Wh=True, mention=True)
            raise ValueError(f"{tlm()} Error loading embedding table from {self.json_table_path}: {e}")
    
    def save_embedding_table(self, embedding_table):
        try:
            with open(self.json_table_path, "w", encoding="latin-1") as f:
                json.dump(self.embedding_table, f, ensure_ascii=False, indent=4)
            self.logger.log(f"Embedding table saved to {self.json_table_path}.", v=True, Wh=True, mention=False)
        except Exception as e:
            self.logger.log(f"Error saving embedding table to {self.json_table_path}: {e}", v=False, Wh=True, mention=True)
            raise ValueError(f"{tlm()} Error saving embedding table to {self.json_table_path}: {e}")

    def create_embedding_model(self):
        # Placeholder for creating the embedding model based on dnn_config
        self.full_model = "DNN Model based on config" # Replace with actual model creation
        self.logger.log("Embedding model created based on DNN configuration.", v=True, Wh=True, mention=False)
    

class model():
    def __init__(self, logger):
        self.logger = logger