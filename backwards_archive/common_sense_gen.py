from transformers import LogitsProcessorList, StoppingCriteriaList, AutoTokenizer, \
    AutoModelWithLMHead
import torch
import torch.nn.functional as F
import numpy as np
from transformers.file_utils import ModelOutput

from backwards_archive.config import ModelConfig, DataConfig, GenerationSpecConfig


def decoder_to_text(output_sequences, tokenizer, input_ids, prompt_text="", reverse=False):
    stop_token = None
    if reverse:
        output_sequences = [torch.from_numpy(np.array(x.tolist()[::-1])) for x in output_sequences]

    generated_sequences = []
    for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
        print("=== GENERATED SEQUENCE {} ===".format(generated_sequence_idx + 1))
        generated_sequence = generated_sequence.tolist()

        # Decode text
        text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
        print(text)
        # Remove all text after the stop token
        text = text[: text.find(stop_token) if stop_token else None]

        # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
        if reverse:
            total_sequence = text[len(tokenizer.decode(input_ids[0], clean_up_tokenization_spaces=True)):]
        else:
            total_sequence = (
                    prompt_text + text[len(tokenizer.decode(input_ids[0], clean_up_tokenization_spaces=True)):]
            )

        generated_sequences.append(total_sequence)

def forward_sample(model,
                   input_ids,
                   mix_rate,
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

    i = 0
    while cur_len < max_length:
        # prepare model inputs
        model_inputs = {"input_ids": input_ids.cuda()}

        # forward pass to get next token
        outputs = model(
            **model_inputs,
            return_dict=True
        )
        next_token_logits = outputs.logits[:, -1, :]

        # if i < model_config.length:
        #     #print(model_config.length-i-1)
        #     #backward_logits[:, :, :]
        #     #res = torch.mean(backward_logits, dim=1)
        #     #perturbed_next_token_logits = next_token_logits + (1 - mix_rate) * res
        #     perturbed_next_token_logits = mix_rate * next_token_logits + (1 - mix_rate) * backward_logits[:, model_config.length-i-1, :]
        # else:
        #     perturbed_next_token_logits = next_token_logits

        perturbed_next_token_logits = next_token_logits

        # pre-process distribution
        next_token_scores = logits_processor(input_ids, perturbed_next_token_logits)
        next_token_scores = logits_warper(input_ids, next_token_scores)

        # sample
        probs = F.softmax(next_token_scores, dim=-1)

        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

        # add code that transforms next_tokens to tokens_to_add
        if eos_token_id is not None:
            assert pad_token_id is not None, "If eos_token_id is defined, make sure that pad_token_id is defined."
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # add token and increase length by one
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        cur_len = cur_len + 1
        i+=1
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


def prepare_model(model,
                  input_ids,
                  max_length=None,
                  min_length=None,
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
                  diversity_penalty=None,
                  output_scores=None,
                  output_attentions=None,
                  output_hidden_states=None,
                  **model_kwargs):
    max_length = max_length if max_length is not None else model.config.max_length
    num_beams = num_beams if num_beams is not None else model.config.num_beams
    pad_token_id = pad_token_id if pad_token_id is not None else model.config.pad_token_id
    bos_token_id = bos_token_id if bos_token_id is not None else model.config.bos_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else model.config.eos_token_id

    output_scores = output_scores if output_scores is not None else model.config.output_scores
    output_attentions = output_attentions if output_attentions is not None else model.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else model.config.output_hidden_states
    )
    decoder_start_token_id = None

    model_kwargs["output_attentions"] = output_attentions
    model_kwargs["output_hidden_states"] = output_hidden_states

    if input_ids is None:
        # init `input_ids` with bos_token_id
        input_ids = model._prepare_input_ids_for_generation(bos_token_id, model_kwargs.get("encoder_outputs"))

    if model_kwargs.get("attention_mask", None) is None:
        # init `attention_mask` depending on `pad_token_id`
        model_kwargs["attention_mask"] = model._prepare_attention_mask_for_generation(
            input_ids, pad_token_id, eos_token_id
        )


    # special case if pad_token_id is not defined
    if pad_token_id is None and eos_token_id is not None:
        pad_token_id = eos_token_id


    if model.config.is_encoder_decoder:
        # add encoder_outputs to model_kwargs
        model_kwargs = model._prepare_encoder_decoder_kwargs_for_generation(input_ids, model_kwargs)

        # set input_ids as decoder_input_ids
        if "decoder_input_ids" in model_kwargs:
            input_ids = model_kwargs.pop("decoder_input_ids")
        else:
            input_ids = model._prepare_decoder_input_ids_for_generation(
                input_ids, decoder_start_token_id=decoder_start_token_id, bos_token_id=bos_token_id
            )

        if "encoder_outputs" not in model_kwargs or not isinstance(model_kwargs["encoder_outputs"], ModelOutput):
            raise ValueError("Make sure that `model_kwargs` include `encoder_outputs` of type `ModelOutput`.")

    if input_ids.shape[-1] >= max_length:
        input_ids_string = "decoder_input_ids" if model.config.is_encoder_decoder else "input_ids"


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
        is_encoder_decoder=False
    )

    config = GenerationSpecConfig()
    config.input_ids = input_ids
    config.max_length = max_length
    config.logits_warper = logits_warper
    config.stopping_criteria = stopping_criteria
    config.logits_processor = logits_processor
    config.model_kwargs = model_kwargs
    config.pad_token_id = pad_token_id
    config.eos_token_id = eos_token_id
    config.bos_token_id = bos_token_id
    return config




if __name__ == '__main__':
    model_config = ModelConfig.from_json("configs/model_config.json")
    data_config = DataConfig.from_json("configs/data_config.json")

    tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-common_gen")
    model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-common_gen")

    model.to(model_config.device)

    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    words = "book write man"

    #input_ids = tokenizer.encode(words, add_special_tokens=False, return_tensors="pt").to('cuda:0')
    features = tokenizer([words], return_tensors='pt')

    num_return_sequences = 5
    forward_generation_spec = prepare_model(model,
                                            input_ids=features['input_ids'].cuda(),
                                            attention_mask=features['attention_mask'].cuda(),
                                            max_length=model_config.max_length + len(features['input_ids'][0]),
                                            temperature=model_config.temperature_forward,
                                            top_k=model_config.top_k_forward,
                                            top_p=model_config.top_p_forward,
                                            repetition_penalty=model_config.repetition_penalty_forward,
                                            num_return_sequences=num_return_sequences)

    output_sequences = forward_sample(
        model,
        forward_generation_spec.input_ids,
        mix_rate=model_config.mix_rate,
        logits_processor=forward_generation_spec.logits_processor,
        logits_warper=forward_generation_spec.logits_warper,
        stopping_criteria=forward_generation_spec.stopping_criteria,
        max_length=forward_generation_spec.max_length,
        pad_token_id=forward_generation_spec.pad_token_id,
        eos_token_id=forward_generation_spec.eos_token_id,
        model_kwargs=forward_generation_spec.model_kwargs
    )

    decoder_to_text(output_sequences,
                    tokenizer,
                    forward_generation_spec.input_ids)