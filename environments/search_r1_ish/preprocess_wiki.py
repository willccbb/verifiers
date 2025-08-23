import Stemmer
import bm25s
from datasets import load_dataset

def main(index_dir="wiki-index-bm25s"):
    ds = load_dataset("wikimedia/wikipedia", "20231101.en")["train"]
    corpus = ds.map(lambda x: dict(combo=f"title:{x['title']} content:{x['text']}"))


    stemmer = Stemmer.Stemmer("english")
    stemmer = Stemmer.Stemmer("english")
    tokenizer = bm25s.tokenization.Tokenizer(stemmer=stemmer)
    corpus_tokens = tokenizer.tokenize([r["combo"] for r in corpus], return_as="tuple")
    retriever = bm25s.BM25(corpus=[r for r in ds], backend="numba")
    retriever.index(corpus_tokens)
    retriever.save(index_dir)
    tokenizer.save_vocab(index_dir)
    tokenizer.save_stopwords(index_dir)
    print("saved index to", index_dir)
    mem_use = bm25s.utils.benchmark.get_max_memory_usage()
    print(f"Peak memory usage: {mem_use:.2f} GB")

if __name__ == '__main__':
    main()
