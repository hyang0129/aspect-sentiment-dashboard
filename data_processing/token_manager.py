from transformers import AutoTokenizer
from tqdm import tqdm
import tensorflow as tf

class Token_Manager:

    def __init__(self, P):

        self.tokenizer = AutoTokenizer.from_pretrained(P.model_name)
        self.P = P

    @staticmethod
    def to_question_answer(example):
        question = ''.join([k + '? ' for k, v in example['aspects'].items() if v != 'none'])
        context = example['text']

        question_context = f"answer_me: {question} context: {context} </s>"

        answer = ', '.join(['%s %s' % (k, v) for k, v in example['aspects'].items() if v != 'none'])
        answer = f"{answer} </s>"

        example['question_and_context'] = question_context
        example['answer'] = answer

        return example


    def encode_to_tokens(self, example, decoder = True):

        encoder_inputs = self.tokenizer(example['question_and_context'], truncation=True,
                                   return_tensors='tf', max_length=self.P.encoder_max_len,
                                   pad_to_max_length=True)

        input_ids = encoder_inputs['input_ids'][0]
        input_attention = encoder_inputs['attention_mask'][0]

        if decoder:
            decoder_inputs = self.tokenizer(example['answer'], truncation=True,
                                       return_tensors='tf', max_length=self.P.decoder_max_len,
                                       pad_to_max_length=True)

            target_ids = decoder_inputs['input_ids'][0]
            target_attention = decoder_inputs['attention_mask'][0]

        else:
            target_ids = tf.zeros((1,))
            target_attention = tf.zeros((1,))

        return {'input_ids': input_ids,
                'attention_mask': input_attention,
                'labels': target_ids,
                'decoder_attention_mask': target_attention}




    def tokenize_dataset(self, examples, compose_qa_fn = None):

        tokenized_dataset = {'input_ids': [],
                             'labels': [],
                             'attention_mask': [],
                             'decoder_attention_mask': []}

        if compose_qa_fn is not None:
            examples = [compose_qa_fn(example) for example in examples]
        else:
            examples = [self.to_question_answer(example) for example in examples]


        for example in tqdm(examples):
            values = self.encode_to_tokens(example)

            for i, k in enumerate(tokenized_dataset.keys()):
                tokenized_dataset[k].append(values[k])

        for k, v in tokenized_dataset.items():
            tokenized_dataset[k] = tf.stack(v, axis=0)

        self.tokenized_dataset = tokenized_dataset
        return tokenized_dataset, examples

    @staticmethod
    def _to_x_none(example):
        return (example, tf.ones((1,)))

    @staticmethod
    def idx_slice_dictionary(dict_, idx):
        try:
            return {k: v[idx] for k, v in dict_.items()}
        except:
            return {k: tf.convert_to_tensor(v.numpy()[idx], dtype=tf.int32) for k, v in dict_.items()}

    def get_dataset(self, index = None, shuffle=False, batch_size=None, repeat=False,
                    to_x_none=True, limit=False):

        if batch_size is None:
            batch_size = self.P.batch_size

        if index is None:
            ds = self.tokenized_dataset
        else:
            ds = self.idx_slice_dictionary(self.tokenized_dataset, index)

        ds = tf.data.Dataset.from_tensor_slices(ds).cache()

        ds = ds.repeat() if repeat else ds
        ds = ds.shuffle(self.P.shuffle_buffer) if shuffle else ds
        ds = ds.map(self._to_x_none) if to_x_none else ds
        ds = ds.batch(batch_size) if batch_size > 0 else ds

        return ds

    @staticmethod
    def strip_special_tokens(text):
        return text.replace('<pad>', '').replace('</s>', '')

    def inference_encode_to_tokens(self, input_text):

        encoded_query = self.tokenizer(input_text,
                                  return_tensors='tf', padding=True, pad_to_max_length=True, truncation=True)

        return encoded_query

    def batch_generate_answer(self, model, examples):

        encoded_query = self.process_question_and_context_for_inference([example['question_and_context'] for example in examples])

        input_ids = encoded_query["input_ids"]
        attention_mask = encoded_query["attention_mask"]

        generated_answer = model.generate(input_ids, attention_mask=attention_mask,
                                        max_length=self.P.decoder_max_len, top_p=0.98, top_k=50)

        answers = []
        for i in range(len(generated_answer.numpy())):
            answers.append(self.strip_special_tokens(self.tokenizer.decode(generated_answer.numpy()[i])))

        for i, example in enumerate(examples):
            examples[i]['generated_answer'] = answers[i]

        return examples