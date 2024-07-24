import time
import torch
import transformers
from openai import OpenAI
client = OpenAI()
import os
import fcntl
from transformers import AutoTokenizer
from src.exp_utils import directories


pipeline = None

class Llama2Wrapper():
    def __init__(self, MODEL_DIR, device):
        global pipeline

        self.device = device

        if pipeline is None:
            self.pipeline = transformers.pipeline(
                "text-generation",
                model=MODEL_DIR,
                torch_dtype=torch.float16,
                device=self.device
            )
            pipeline = self.pipeline
        else:
            print('using preloaded llama')
            self.pipeline = pipeline
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    def generate(self, prompt, max_new_tokens=1000):
        prompt = prompt.replace('[APE]', '').strip()

        response = self.__generate_text(prompt, max_new_tokens=max_new_tokens)

        return response

    def __generate_text(self, prompt, max_new_tokens=1000):
        sequences = self.pipeline(
            prompt,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            # max_length=1000,
            max_new_tokens=max_new_tokens,
        )

        output = ''
        for seq in sequences:
            # print(f"Result: {seq['generated_text']}")
            output += seq['generated_text'] + '\n'

        assert output[:len(prompt)] == prompt
        response = output[len(prompt):]
        return response


class GPT2():
    def __init__(self, name):
        from transformers import GPT2Tokenizer, GPT2Model
        self.tokenizer = GPT2Tokenizer.from_pretrained(name, cache_dir=directories.cache_dir)
        self.model = GPT2Model.from_pretrained(name, cache_dir=directories.cache_dir).cuda()

    def get_embedding(self, text):
        """
            return shape: [batch_size=1, seq_len, dim]
        """
        with torch.no_grad():
            encoded_input = self.tokenizer(text, return_tensors='pt')
            encoded_input['input_ids'] = encoded_input['input_ids'].cuda()
            encoded_input['attention_mask'] = encoded_input['attention_mask'].cuda()
            output = self.model(**encoded_input)
        
        return output[0].detach().cpu()


class OpenAIModels():
    def __init__(self, model="text-embedding-ada-002", benchmark='none'):
        self.model = model
        self.benchmark = benchmark

        self.load_saved_embeddings()

        self.cnt = 0

    def load_saved_embeddings(self):
        self.path = f'../cache/ada_embeddings_{self.benchmark}.pth'
        
        print('loading saved embedding')
        if os.path.exists(self.path):
            self.embedding_cache = torch.load(self.path)
        else:
            self.embedding_cache = {}
        print('done')

    def get_embedding(self, text):
        text = text.replace("\n", " ")
        if text == '':
            text = ' '

        ## memoization
        if text in self.embedding_cache:
            return self.embedding_cache[text]

        ret = None
        while ret is None:
            try:
                response = client.embeddings.create(input=[text],
                            model=self.model).data[0].embedding
                ret = torch.tensor(response).unsqueeze(0).unsqueeze(0)

            except Exception as e:
                print(e)
                if 'rate limit' in str(e).lower():  ## rate limit exceed
                    print('wait for 10s and retry...')
                    time.sleep(10)
                else:
                    print('Retrying...')
                    time.sleep(2)
    
        self.embedding_cache[text] = ret

        self.cnt += 1
        if self.cnt % 25 == 0:
            torch.save(self.embedding_cache, self.path)

        return ret

    def _write_embedding_safely(self):
        torch.save(self.embedding_cache, self.path)


class SentenceT5():
    def __init__(self, name):
        from sentence_transformers import SentenceTransformer
        self.name = f'sentence-{name}'
        self.model = SentenceTransformer(f'sentence-transformers/{self.name}', cache_folder=directories.cache_dir)

        self._load_saved_embeddings()

    def _load_saved_embeddings(self):
        self.path = f'../cache/{self.name}_embeddings.pth'
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        
        if os.path.exists(self.path):
            self.embedding_cache = torch.load(self.path)
        else:
            self.embedding_cache = {}

    def get_embedding(self, sentences):
        if not isinstance(sentences, list):
            sentences = [sentences]

        assert len(sentences) == 1
        
        if sentences[0] in self.embedding_cache:
            return self.embedding_cache[sentences[0]]

        embeddings = self.model.encode(sentences)
        embeddings = torch.tensor(embeddings).unsqueeze(0)

        self.embedding_cache[sentences[0]] = embeddings
        self._write_embedding_safely()

        return embeddings

    def _write_embedding_safely(self):
        with open(self.path, 'a') as file:
            try:
                fcntl.flock(file, fcntl.LOCK_EX)  # Acquire an exclusive lock
                torch.save(self.embedding_cache, self.path)
            finally:
                fcntl.flock(file, fcntl.LOCK_UN)  # Release the lock
