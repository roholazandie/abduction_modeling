from transformers import GPT2LMHeadModel, GPT2Tokenizer, LogitsProcessorList, StoppingCriteriaList
from transformers.generation_utils import GenerationMixin
import torch
import torch.nn.functional as F
import numpy as np
from config import ModelConfig, DataConfig
from utils import load_dataset


def sample(model,
           input_ids,
           logits_processor,
           logits_warper,
           stopping_criteria,
           max_length,
           pad_token_id,
           eos_token_id,
           model_kwargs):
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    max_length = max_length if max_length is not None else model.config.max_length
    logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
    pad_token_id = pad_token_id if pad_token_id is not None else model.config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else model.config.eos_token_id

    # init sequence length tensors
    sequence_lengths, unfinished_sequences, cur_len = model._init_sequence_length_for_generation(
        input_ids, max_length
    )
    scores = None

    while cur_len < max_length:
        # prepare model inputs
        model_inputs = model.prepare_inputs_for_generation(input_ids)

        # forward pass to get next token
        outputs = model(
            **model_inputs,
            return_dict=True
        )
        next_token_logits = outputs.logits[:, -1, :]

        # pre-process distribution
        next_token_scores = logits_processor(input_ids, next_token_logits)
        next_token_scores = logits_warper(input_ids, next_token_scores)

        # sample
        probs = F.softmax(next_token_scores, dim=-1)
        np.save('my.npy', probs.cpu().numpy())

        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

        # add code that transforms next_tokens to tokens_to_add
        if eos_token_id is not None:
            assert pad_token_id is not None, "If eos_token_id is defined, make sure that pad_token_id is defined."
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # add token and increase length by one
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        cur_len = cur_len + 1

        # update sequence length
        if eos_token_id is not None:
            sequence_lengths, unfinished_sequences = model._update_seq_length_for_generation(
                sequence_lengths, unfinished_sequences, cur_len, next_tokens == eos_token_id
            )

        # stop when there is a </s> in each sentence, or if we exceed the maximum length
        if unfinished_sequences.max() == 0:
            break

        if stopping_criteria(input_ids, scores):
            break

        # update model kwargs
        model_kwargs = model._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=model.config.is_encoder_decoder
        )

    return input_ids


def generate(model,
             input_ids,
             max_length=None,
             min_length=None,
             do_sample=None,
             early_stopping=None,
             num_beams=None,
             temperature=1.0,
             top_k=0,
             top_p=None,
             repetition_penalty=1.0,
             bos_token_id=None,
             pad_token_id=None,
             eos_token_id=None,
             length_penalty=None,
             num_return_sequences=None,
             diversity_penalty=None):
    max_length = max_length if max_length is not None else model.config.max_length
    num_beams = num_beams if num_beams is not None else model.config.num_beams
    pad_token_id = pad_token_id if pad_token_id is not None else model.config.pad_token_id
    bos_token_id = bos_token_id if bos_token_id is not None else model.config.bos_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else model.config.eos_token_id

    # special case if pad_token_id is not defined
    if pad_token_id is None and eos_token_id is not None:
        pad_token_id = eos_token_id

    # get distribution pre_processing samplers
    logits_processor = model._get_logits_processor(
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=None,
        encoder_no_repeat_ngram_size=None,
        encoder_input_ids=None,
        bad_words_ids=None,
        min_length=min_length,
        max_length=max_length,
        eos_token_id=eos_token_id,
        forced_bos_token_id=None,
        forced_eos_token_id=None,
        prefix_allowed_tokens_fn=None,
        num_beams=num_beams,
        num_beam_groups=None,
        diversity_penalty=diversity_penalty,
        remove_invalid_values=None,
    )

    stopping_criteria = model._get_stopping_criteria(max_length=max_length, max_time=None)

    logits_warper = model._get_logits_warper(
        top_k=top_k, top_p=top_p, temperature=temperature, num_beams=num_beams
    )

    # expand input_ids with `num_return_sequences` additional sequences per batch
    input_ids, model_kwargs = model._expand_inputs_for_generation(
        input_ids,
        expand_size=num_return_sequences,
        is_encoder_decoder=model.config.is_encoder_decoder,
    )

    return sample(
        model,
        input_ids,
        logits_processor=logits_processor,
        logits_warper=logits_warper,
        stopping_criteria=stopping_criteria,
        max_length=max_length,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        model_kwargs=model_kwargs
    )


def generate_abductive_explanation(model_config,
                                   forward_model,
                                   forward_tokenizer,
                                   backward_model,
                                   backward_tokenizer,
                                   o1_text,
                                   o2_text):
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    prompt_text = "Ray was fine but his car was totaled."
    input_ids = forward_tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt").to('cuda:0')
    length = 10
    repetition_penalty = 1.0
    num_return_sequences = 5
    stop_token = None
    output_sequences = generate(forward_model,
                                input_ids,
                                max_length=length + len(input_ids[0]),
                                temperature=1.0,
                                top_k=0,
                                top_p=0.9,
                                repetition_penalty=repetition_penalty,
                                do_sample=True,
                                num_return_sequences=num_return_sequences,
                                )

    generated_sequences = []
    for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
        print("=== GENERATED SEQUENCE {} ===".format(generated_sequence_idx + 1))
        generated_sequence = generated_sequence.tolist()

        # Decode text
        text = forward_tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

        # Remove all text after the stop token
        text = text[: text.find(stop_token) if stop_token else None]

        # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
        total_sequence = (
                prompt_text + text[len(forward_tokenizer.decode(input_ids[0], clean_up_tokenization_spaces=True)):]
        )

        generated_sequences.append(total_sequence)
        print(total_sequence)

    return


if __name__ == '__main__':
    model_config = ModelConfig.from_json("configs/model_config.json")
    data_config = DataConfig.from_json("configs/data_config.json")

    forward_model = GPT2LMHeadModel.from_pretrained(model_config.pretrained_model_forward)
    forward_model.to(model_config.device)
    forward_model.eval()
    forward_tokenizer = GPT2Tokenizer.from_pretrained(model_config.pretrained_model_forward)

    # Freeze GPT-2 weights
    for param in forward_model.parameters():
        param.requires_grad = False

    backward_model = GPT2LMHeadModel.from_pretrained(model_config.pretrained_model_backward)
    backward_model.to(model_config.device)
    backward_model.eval()
    backward_tokenizer = GPT2Tokenizer.from_pretrained(model_config.pretrained_model_backward)

    # Freeze GPT-2 weights
    for param in backward_model.parameters():
        param.requires_grad = False

    records = load_dataset(data_config.abductive_dataset)
    print(records)

    for r in records:
        o1_text = '<|endoftext|>'.join([r['obs2'], r['obs1']])
        o2_text = r['obs2']

        text = generate_abductive_explanation(model_config,
                                              forward_model,
                                              forward_tokenizer,
                                              backward_model,
                                              backward_tokenizer,
                                              o1_text,
                                              o2_text)
        print("final", text)
