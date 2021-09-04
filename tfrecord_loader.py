import jax
import transformers
import zstandard as zstd
import jsonlines
import io
import subprocess
import shlex
import tensorflow as tf
import numpy as np
from google.cloud import storage
class WriteJsonlzst:
    """
    Writes (text) file streams with jsonl.zst extension into tfrecords

    Attributes:
    
        encoder: The huggingface encoder to utilize for encoding the text
        split_size: Length of each tokenized tensor
    """
    def __init__(self,encoder=None,split_size=2048):
        if(encoder is None):
            encoder = transformers.GPT2TokenizerFast.from_pretrained('gpt2',max_length=1e9)
        self.encoder = encoder
        self.split_size = split_size

    def _decompress_to_serialized_text(self,jsonl_zst_stream,key='text',para_joiner = '\n\n'):
        """
        Returns a serialized tokenized text generator from jsonl.zst streams
        """
        cctx = zstd.ZstdDecompressor()
        reader = io.BufferedReader(cctx.stream_reader(jsonl_zst_stream))
        rdr = jsonlines.Reader(reader) #Creating a complex text retrival stream

        tokens = tf.constant([],dtype=tf.int32) #tokens of text from stream
        
        for ob in rdr:#iterating through text
            
            text = ob[key]
            if isinstance(text, list):
                text = para_joiner.join(text)
            text += ' <|endoftext|>'


            tokens = tf.concat([tokens,
                                self.encoder(text,return_tensors='tf',return_attention_mask=False).input_ids[0]],
                                axis = 0) #convert text into tokens and utilize previously left tokens
            
            splits = [self.split_size for i in range(self.split_size,len(tokens),self.split_size)] 
                #create a splits array whose length is equal to the number of {self.split_size} sized arrays

            if(len(splits) > 0):
                splits.append(len(tokens)-len(splits)*self.split_size) 
                    #adding the left over tokens length to correctly split the array

                seq= tf.split(tokens,splits)
                data,tokens = seq[:-1],seq[-1]
                    #splitting and reusing to leftover tokens in next iteration (if present)

                for i in data:
                    yield tf.io.serialize_tensor(i).numpy()
                        #serialize and return data
    def processlink(self,link):
        """
        Converts the url link to Google cloud storage path and checks if the path exists.
        If the file doesn't exist, it is created

        Returns the Google cloud storage path for the file
        """
        bucket_name = 'gpt-guru'
        train = link.rfind('train') != link.find('train') 
        #Code to find if the link references a training record.
        #Very specific to https://the-eye.eu/public/AI/training_data/code_clippy_data/code_clippy_dedup_data/
        #Change it to reflect your training data location

        path = f"gs://{bucket_name}/{'train' if train else 'test'}/{link[link.rfind('/')+1:-10]}.tfrecords" #get appropriate path
        
        
        storage_client = storage.Client()
        
        bucket = storage_client.bucket(bucket_name)
        stats = storage.Blob(bucket=bucket, name=path[6+len(bucket_name):]).exists(storage_client)
        #Check if the path exists

        if(stats or link[:2] == 'gs'):
            #Path exists. (or it is a gs path)return it
            return path
        else:
            #Path doesn't exist. Create one and return it

            command = "wget" + " -O -" + " "+ link
            command = shlex.split(command)
            res = subprocess.Popen(command,stdout=subprocess.PIPE)
            out,err = res.communicate()
            while(res.poll() is None):
                pass
            #Download the file

            with tf.io.TFRecordWriter(path) as writer:
                self._write(io.BytesIO(out),writer)
            #Create the tfrecord

            return path

    def _write(self,jsonl_zst_stream,writer):
        """
        Writes text from jsonl_zst_stream
        """ 
        for i in self._decompress_to_serialized_text(jsonl_zst_stream):
            writer.write(i)
class TFRecordLoader:
    def __init__(self, index_fname, batch_size, parse_fn, map_fn=None, restore_state=None):
        self.writer = WriteJsonlzst() #addition: to check and create files at runtime
        if restore_state is not None:
            self.file_idx = restore_state["file_idx"]
            self.file_idx_init = False
            self.used = restore_state["used"]
        else:
            self.file_idx = 0
            self.file_idx_init = True
            self.used = []

        self.index = open(index_fname).read().splitlines()
        self.clean_index = list(filter(lambda x: x not in self.used, self.index))
        self.bs = batch_size
        # self.seq = sample_size
        self.parse_fn = parse_fn

        if map_fn:
            self.map_fn = map_fn
        else:
            self.map_fn = lambda x: x

        self.sample_fn = self.sample_once()

    def reset(self):
        self.file_idx = 0
        self.file_idx_init = True
        self.used = []

        self.clean_index = list(filter(lambda x: x not in self.used, self.index))
        self.sample_fn = self.sample_once()

    def sample_once(self):
        for i in self.clean_index:
            compression = "ZLIB" if "zstd" in i else ""
            i = self.writer.processlink(i) #Gets Google cloud storage path from the url link
            file = tf.data.TFRecordDataset(i, compression_type=compression).map(self.parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
            file = file.apply(tf.data.experimental.dense_to_ragged_batch(np.prod(self.bs), drop_remainder=True))
            file = file.prefetch(10)

            for file_idx, data in enumerate(file):
                data = jax.tree_map(lambda x: x.numpy(), data)
                data = self.map_fn(data)

                if not self.file_idx_init and file_idx <= self.file_idx:
                    if file_idx % 1000 == 0:
                        print(f"skipping to batch {self.file_idx}, currently at {file_idx}")
                    continue
                self.file_idx_init = True
                self.file_idx = file_idx
                yield jax.tree_map(lambda x: x.reshape(self.bs + x.shape[1:]), data)
            self.used.append(i)
            self.file_idx = 0

    # this loops infinitely, use .sample_once to get an iterator for validation
    def get_samples(self):
        try:
            return next(self.sample_fn)
        except StopIteration:
            self.reset()
            return self.get_samples()

    def get_state(self):
        return {
            "used": self.used,
            "file_idx": self.file_idx
        }


class TFRecordNewInputs(TFRecordLoader):
    def __init__(self, index_fname, batch_size, sample_size, restore_state=None):
        def tf_parse(example_proto):
            return tf.io.parse_tensor(example_proto,out_type=tf.int32)
        super().__init__(index_fname, batch_size, tf_parse, restore_state=restore_state)


class TFRecordWIT(TFRecordLoader):
    def __init__(self, index_fname, batch_size, restore_state=None, text_tokens=256):
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.tokenizer.pad_token = "<|endoftext|>"
        self.tokenizer.add_special_tokens({'sep_token': '<|sep|>', 'pad_token': '<|pad|>'})

        def map_fn(example):
            tokenizer = self.tokenizer

            def decode(x):
                return tokenizer(["<|endoftext|>" + i.decode() for i in x])["input_ids"]

            texts = [
                decode(example["context_page_description"]),
                decode(example["context_section_description"]),
                decode(example["caption_reference_description"]),
                decode(example["caption_alt_text_description"]),
                decode(example["caption_attribution_description"]),
            ]

            output = []

            for text, dalle in zip(zip(*texts), example["dalle"]):
                all_text = list(itertools.chain(*text))[-text_tokens+1:]

                all_text += [tokenizer.pad_token_id] * ((text_tokens - 1) - len(all_text))

                assert len(all_text) == text_tokens - 1

                all_tokens = all_text + [tokenizer.sep_token_id] + list(dalle + tokenizer.vocab_size + 1)
                output.append(all_tokens)

            return np.array(output)

        def tf_parse(example_proto):
            features = {
                "page_title": tf.io.FixedLenFeature([], tf.string),
                "section_title": tf.io.FixedLenFeature([], tf.string),
                "hierarchical_section_title": tf.io.FixedLenFeature([], tf.string),
                "caption_reference_description": tf.io.FixedLenFeature([], tf.string),
                "caption_attribution_description": tf.io.FixedLenFeature([], tf.string),
                "caption_alt_text_description": tf.io.FixedLenFeature([], tf.string),
                "mime_type": tf.io.FixedLenFeature([], tf.string),
                "context_page_description": tf.io.FixedLenFeature([], tf.string),
                "context_section_description": tf.io.FixedLenFeature([], tf.string),

                "dalle": tf.io.FixedLenFeature([1024], tf.int64),
            }

            parsed_features = tf.io.parse_single_example(example_proto, features)

            return parsed_features

        super().__init__(index_fname, batch_size, tf_parse, map_fn, restore_state=restore_state)


if __name__ == "__main__":
    # d = TFRecordNewInputs("data/pile.val.index", (8, 32), 2048)
    # for idx, i in enumerate(d.sample_once()):
    #     print(i)
    #     break

    d = TFRecordWIT("data/wit_dalle.train.index", (8, 32))
    for idx, i in enumerate(d.sample_once()):
        print(i)
        break

    print()
