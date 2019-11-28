import numpy as np
from .constants import W2V_EMBED_SIZE

class NLP():

    def __init__(self, w2v={}):
        self.mem_embed = {}
        self.w2v = w2v

    # TODO: Look into methods with all uppercase tokens
    # TODO: Use fixed (random) embeddings for common tokens (e.g. ETH, CEO, ICO)
    def tokenize_method(self, method):
        tokens = []
        for tmp in method.split('_'):
            if tmp.isupper():
                tokens.append(tmp.lower())
                continue
            j = 0
            for i in range(len(tmp)):
                if tmp[i].isupper():
                    tokens.append(tmp[j:i].lower())
                    j = i
            tokens.append(tmp[j:].lower())
        return tokens

    def embed_token(self, token):
        if token not in self.w2v:
            return None #self.w2v[token] = np.random.randn(W2V_EMBED_SIZE)
        return self.w2v[token]
    
    def embed_method(self, method):
        if method in self.mem_embed:
            return self.mem_embed[method]
        
        tokens = self.tokenize_method(method)
        embeds = []
        for token in tokens:
            token_embed = self.embed_token(token)
            if token_embed is None:
                continue
            embeds.append(token_embed.reshape(1,-1))
        if len(embeds) == 0:
            self.mem_embed[method] = np.zeros(W2V_EMBED_SIZE)
        else:
            self.mem_embed[method] = np.mean(np.concatenate(embeds, axis=0), axis=0).reshape(-1)
        return self.mem_embed[method]


